library(tidyverse)
library(gridExtra)

theme_fontsize <- theme(text = element_text(size=10),
          axis.text.x = element_text(size=10), axis.text.y = element_text(size=10),
          legend.text = element_text(size=10), legend.title=element_text(size=10),
          strip.text.x = element_text(size=10), strip.text.y = element_text(size=10))

idir <- "results/"
ifile.list <- list.files(idir)

results <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
})) %>%
    filter(n %in% c(100,150,200,300,500,1000,2000,3000,5000))
    

df <- results %>%
    filter(Symmetry==0, Method != "CQR2") %>%
    gather(`Coverage`, `Conditional coverage`, `Length`, `Length cover`, key="key", value="value") %>%
    group_by(Method, Alpha, n, Symmetry, key) %>%
    summarise(Skewness=mean(Skewness), value.se = 2*sd(value)/sqrt(n()), value = mean(value), N=n()) %>%
    ungroup()

p1 <- df %>%
    filter(key %in% c("Coverage", "Conditional coverage"), Method!="Oracle") %>%
    mutate(key = factor(key, c("Coverage", "Conditional coverage"), c("Marginal", "Conditional"))) %>%
    ggplot(aes(x=n, y=value, color=Method, shape=Method)) +
    geom_hline(aes(yintercept=1-Alpha), linetype=2) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=value-value.se, ymax=value+value.se), alpha=0.5) +
    facet_grid(.~key, scales="free") +
    ylim(0.85,0.95) +
    scale_x_continuous(trans="log10") +
    guides(color=guide_legend(ncol=2)) +
    theme_bw() +
    xlab("Sample size") +
    ylab("Coverage")  +
    theme(legend.position = "none") +
    theme_fontsize
p1

df.oracle <- df %>%
    filter(Method=="Oracle", key=="Length")
length.oracle <- mean(df.oracle$value)

p2 <- df %>%
    filter(key %in% c("Length"), Method!="Oracle") %>%
    mutate(key = factor(key, c("Length", "Length cover"), c("Width", "Conditional on coverage"))) %>%
    ggplot(aes(x=n, y=value, color=Method, shape=Method)) +
    geom_hline(aes(yintercept=length.oracle), linetype=2) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=value-value.se, ymax=value+value.se), alpha=0.5) +
    facet_grid(.~key, scales="free") +
    ylim(3,NA) +
    scale_x_continuous(trans="log10") +
    xlab("Sample size") +
    ylab("Width") +
    theme_bw() +
    theme(legend.position = "right") +
    theme_fontsize
p2


pp <- grid.arrange(p1, p2, widths=c(1.8,1.5), ncol=2)
pp %>% ggsave(file="exp_synthetic_n.png", height=2, width=7, units = "in")


df <- results %>%
    filter(n==1000, Method != "CQR2") %>%
    gather(`Coverage`, `Conditional coverage`, `Length`, `Length cover`, key="key", value="value") %>%
    group_by(Method, Alpha, n, Symmetry, key) %>%
    summarise(Skewness=mean(Skewness), value.se = 2*sd(value)/sqrt(n()), value = mean(value)) %>%
    ungroup()

p1 <- df %>%
    filter(key %in% c("Coverage", "Conditional coverage"), Method!="Oracle") %>%
    mutate(key = factor(key, c("Coverage", "Conditional coverage"), c("Marginal", "Conditional"))) %>%
    mutate(Symmetry = Symmetry*2/100) %>%
    ggplot(aes(x=Skewness, y=value, color=Method, shape=Method)) +
    geom_hline(aes(yintercept=1-Alpha), linetype=2) +
    geom_point() +
    geom_line() +
    geom_errorbar(aes(ymin=value-value.se, ymax=value+value.se), alpha=0.5) +
    facet_grid(.~key, scales="free") +
    ylim(0.85,0.95) +
    scale_x_continuous(breaks=c(0:3), limits=c(0,3)) +
    theme_bw() +
    xlab("Skewness") +
    ylab("Coverage") +
    theme(legend.position = "none") +
    theme_fontsize
p1


df.oracle <- df %>%
    mutate(Method = ifelse(Method=="Oracle", NA, Method)) %>%
    filter(is.na(Method), key=="Length") %>%
    mutate(key = "Width")
length.oracle <- mean(df.oracle$value)

p2 <- df %>%
    filter(key %in% c("Length"), Method!="Oracle") %>%
    mutate(key = factor(key, c("Length", "Length cover"), c("Width", "Conditional on coverage"))) %>%
    mutate(Symmetry = Symmetry*2/100) %>%
    ggplot(aes(x=Skewness, y=value, color=Method, shape=Method)) +
    geom_point() +
    geom_line() +
    geom_line(data=df.oracle, aes(x=Skewness, y=value), color="black", linetype=2, show_guide=FALSE) +
    geom_errorbar(aes(ymin=value-value.se, ymax=value+value.se), alpha=0.5) +
    facet_grid(.~key, scales="free") +
    ylim(3,NA) + 
    scale_x_continuous(breaks=c(0:3), limits=c(0,3)) +
    xlab("Skewness") +
    ylab("Width") +
    theme_bw() +
    theme(legend.position = "right") +
    theme_fontsize
p2


pp <- grid.arrange(p1, p2, widths=c(1.8,1.5), ncol=2)
pp %>% ggsave(file="exp_synthetic_symmetry.png", height=2, width=7, units = "in")
