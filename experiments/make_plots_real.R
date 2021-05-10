library(tidyverse)
library(gridExtra)
library(kableExtra)

theme_fontsize <- theme(text = element_text(size=10),
          axis.text.x = element_blank(), axis.text.y = element_text(size=10),
          legend.text = element_text(size=10), legend.title=element_text(size=10),
          strip.text.x = element_text(size=10), strip.text.y = element_text(size=10))

idir <- "results_real/"
ifile.list <- list.files(idir)

results <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))    

method.values <- c("CHR", "CQR", "CQR2", "DistSplit")
method.labels <- c("CHR", "CQR", "DCP-CQR", "DistSplit")

df <- results %>%
    gather(Coverage, `Conditional coverage`, Length, `Length cover`, key="Key", value="Value") %>%
    group_by(Dataset, Method, Box, n_train, n_cal, n_test, Key) %>%
    mutate(Method = factor(Method, method.values, method.labels))

data.values <- c("bio", "blog_data", "facebook_1", "facebook_2", "meps_19", "meps_20", "meps_21")
data.labels <- c("bio", "blog", "fb1", "fb2", "meps19", "meps20", "meps21")

bbox <- "NNet"

p1 <- df %>%
    mutate(Dataset = factor(Dataset, data.values, data.labels)) %>%
    filter(Box==bbox, Method!="DCP") %>%
    filter(Key %in% c("Conditional coverage")) %>%
    ggplot(aes(y=Value, color=Method)) +
    geom_boxplot() +
    facet_grid(.~Dataset) +
    ylab("Conditional coverage") +
    geom_hline(yintercept=0.9, linetype=2) +
    ylim(0.75,1) +
    theme_bw() +    
    theme(legend.position = "none") +
    theme_fontsize
p2 <- df %>%
    mutate(Dataset = factor(Dataset, data.values, data.labels)) %>%
    filter(Box==bbox, Method!="DCP") %>%
    filter(Key %in% c("Length")) %>%
    group_by(Dataset, Box, n_train, n_cal, n_test, Key) %>%
    mutate(Value = Value/min(Value)) %>%
    ggplot(aes(x=Method, y=Value, color=Method)) +
    geom_boxplot() +
    facet_grid(.~Dataset) +
    ylim(1,ifelse(bbox=="NNet", 2.75, 3.75)) +
    ylab("Width (relative)") +    
    theme_bw() +    
    theme(legend.position = "bottom") +
    theme_fontsize
p3 <- cowplot::get_legend(p2)
p2 <- p2 + theme(legend.position = "none")
pp <- cowplot::plot_grid(p1, p2, align = "v", ncol = 1, rel_heights = c(1,1,0.2))

pp2 <- grid.arrange(pp, p3, heights=c(6,1), ncol=1)
pp2 %>% ggsave(file=sprintf("exp_real_%s.png", bbox), height=4, width=7, units = "in")


#################
## Make tables ##
#################

key.values = c("Coverage", "Conditional coverage", "Length", "Length cover")
key.labels = c("Coverage", "Conditional coverage", "Length", "Length cover")

method.values <- c("CHR", "CQR", "DCP", "CQR2", "DistSplit")
method.labels <- c("CHR", "CQR", "DCP", "DCP-CQR", "DistSplit")

df <- results %>%
    gather(Coverage, `Conditional coverage`, Length, `Length cover`, key="Key", value="Value") %>%
    group_by(Dataset, Method, Box, n_train, n_cal, n_test, Key) %>%
    mutate(Method = factor(Method, method.values, method.labels))

tb <- df %>% 
    mutate(Dataset = factor(Dataset, data.values, data.labels)) %>%
    group_by(Dataset, Method, Box, n_train, n_cal, n_test, Key) %>%
    summarise(Mean = mean(Value), SD = sd(Value)) %>%
    ungroup() %>%
    mutate(Value = ifelse(Key=="Length", sprintf("%.1f (%.1f)", Mean, SD), sprintf("%.2f (%.2f)", Mean, SD))) %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    arrange(Dataset, Method, Box, Key) %>%
    select(-Mean, -SD, -n_train, -n_cal, -n_test) %>%
    pivot_wider(values_from=c("Value"), names_from=c("Key", "Box"))
    

tb1 <- tb %>%
    select(-`Length cover_NNet`, -`Length cover_RF`) %>%
    kbl("latex", align="c", booktabs=TRUE,
        col.names = c("Data", "Method",
                      "Marginal", "Condit.", "Width",
                      "Marginal", "Condit.", "Width")) %>%
    collapse_rows(columns = 1, valign = "top", latex_hline="major", longtable_clean_cut=TRUE) %>%
#    pack_rows(index = table(tb$Dataset)) %>%
    add_header_above(c(" " = 2, "Coverage" = 2, " "=1, "Coverage" = 2, " " = 1)) %>%
    add_header_above(c(" " = 2, "Neural Network" = 3, "Random Forest"=3))
    

tb1 %>% save_kable(sprintf("../../paper/tables/data_real.tex"), keep_tex=TRUE, self_contained=FALSE)
