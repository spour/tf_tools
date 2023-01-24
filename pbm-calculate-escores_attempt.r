input_data_path <- ""
data <- read.table(input_data_path, as.is = T, sep = "\t",
                   col.names = 
                     c("#id_spot", "row","col", "control", "id_probe",
"pbm_sequence", "linker_sequence",
"mean_signal_intensity", "mean_background_intensity","flag"))

positives_ranksum = 0
negatives_ranksum = 0
total_top_half_ranksum = 0
num_pos_in_the_top_half = 0
num_pos = 0
num_neg = 0
escore = 0
pos_ranks = c()
kmer <- "ACTGCGCA"
kmer_rev_comp <- stringi::stri_reverse(chartr("ACTG", "TGAC", kmer))

data_sorted_ranked <- data %>% 
  mutate(rank = rank(-mean_signal_intensity, ties.method = "average"),
         kmer_stat = ifelse(grepl(kmer, pbm_sequence) |
                              grepl(kmer_rev_comp, pbm_sequence), "f", "b")) %>%
  arrange(desc(mean_signal_intensity))

sorted_pos_ranks <- data_sorted_ranked %>% filter(kmer_stat == "f")
total_pos_ids <- nrow(sorted_pos_ranks)

# Take only the top half of the positives
num_pos <- ceiling(total_pos_ids/2)

# Calculate the ranksum of the top half of the positives
num_pos_in_the_top_half = 0
total_top_half_ranksum = 0
total_num_probes = nrow(data)

sorted_pos_ranks_arr <- sorted_pos_ranks$rank
total_top_half_probes <- total_num_probes/2
for (j in 1:num_pos){
  if (sorted_pos_ranks_arr[j] < total_top_half_probes){
    positives_ranksum <- positives_ranksum + sorted_pos_ranks_arr[j]
    num_pos_in_the_top_half <- num_pos_in_the_top_half + 1
  }
}

num_pos_in_the_bottom_half <- num_pos - num_pos_in_the_top_half

for (j in 1:num_pos_in_the_bottom_half){
  positives_ranksum <- positives_ranksum + total_top_half_probes + j
}

# Calculate the total ranksum of the top half
total_top_half_ranksum <- ((total_top_half_probes+num_pos_in_the_bottom_half) * (total_top_half_probes + num_pos_in_the_bottom_half + 1))/2

# Calculate the approximate ranksum of the negatives
negatives_ranksum = total_top_half_ranksum - positives_ranksum

num_neg <- total_top_half_probes + num_pos_in_the_bottom_half - num_pos

# From Berger et al. 2006 supplementary
escore = (1/(num_neg + num_pos)) * 
  ((negatives_ranksum/num_neg) - (positives_ranksum/num_pos))

escore

rank_intensities <- function(arrayref){
  n <- length(arrayref)
  corr_factor <- 0
  
  # Index starts at 0 (changed from j=1, but when using j
  # as the rank, add 1) -- modified to be starting from 1 in R
  j=1
  while(j <= n){
    # If the next intensity value is not the same as the current
    # one, assign it the current rank     
    if (arrayref[j + 1] != arrayref[j]){
      arrayref[j] <- j + 1
      j <- j + 1
    }else{
      # Next value is the same as the current, find out how far
      # down the list these ties go
      jt <- j + 1
      while((jt <= n) & (arrayref[jt] == arrayref[j])){
        jt <- jt + 1
      }
      # Ties have now stopped, get the average rank (Using $j and $jt here for rank
      # positions, must add 1 to both)
      rank <- 0.5*(j + 1 + jt + 1 - 1)
      # Assign this rank to all the ties
      ji <- j
      while(ji <= (jt - 1)){
        arrayref[ji] <- rank
        ji <- ji + 1
      }
      t <- jt + 1 - j + 1
      corr_factor <- corr_factor + t * t * t - t
      j=jt
    }
  }
  if (j == n){
    # Add 1 since the position at the end is the last rank and index started at 0
    arrayref[j] <- n
  }
  corr_factor
}



