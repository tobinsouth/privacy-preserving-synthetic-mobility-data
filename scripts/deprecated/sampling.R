library(data.table)
library(bit64)
library(dplyr)
library(tidyverse)
stays=readRDS("data/stays2_31080.Rds")
stays = as.data.table(stays)


sampled_data <- stays %>% 
    group_by(user) %>% 
    nest()  %>% 
    ungroup() %>% 
    slice_sample(n=500)  %>% 
    unnest()
    
write.csv(stays, file=gzfile("data/stays31080.csv.gz"))


