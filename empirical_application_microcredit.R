rm(list = ls())
##------------------------------------------------------------------------------
## Libraries and source files
##------------------------------------------------------------------------------
library(dplyr) # for data manipulation
library(ggplot2) # for plotting
library(xgboost) # gradient boosting for model training
library(RColorBrewer) # for color palettes
## Source file
source( "functions.R" ) 
## Set colors
cb_colors = brewer.pal(n = 8, name = "Set1") # discrete colorblind palette
##------------------------------------------------------------------------------
## Load data and setup variables
##------------------------------------------------------------------------------
## Load csv file
df= read.csv("microcredit.csv")
## Compute assignment probabilities for each stratum
df = df %>% mutate(S=aimag) %>% # "aimag" means province, rename it as S for stratum variable
  group_by(S) %>%
  mutate(prob_treatment = mean(W)) %>% # compute assignment probability within each stratum
  ungroup()
## Select outcome variable to consider
outcome_name = "entrevenue"
outcome_title = "Enterprise revenue"
df= df %>% mutate(Y=entrevenue_f/1000) # rescale by dividing by 1000
df = df %>% filter(!(is.na(Y))) # remove data with missing outcome variable
## Select pre-treatment covariates
covariate_names = c('hhincome', 'entrevenue', 'hhwageinc',
                    'hhsize', 'eduhigh', 'age',
                    'age_sq', 'buddhist', 'under16',
                    'eduvoc', 'hahl', 'loan_baseline',
                    'marr_cohab', 'age16m', 'age16f', 'entprofit') 
##------------------------------------------------------------------------------
## Estimation setup
##------------------------------------------------------------------------------
## Locations for DTE and PTE estimation
vec.loc.min = 0
vec.loc.max = 200
bin.width = 10
## Locations for DTE
vec.loc = seq(vec.loc.min, vec.loc.max, by = bin.width) 
## Locations for PTE
vec.loc.up = vec.loc
vec.loc.low = c(vec.loc.min-1, vec.loc[-length(vec.loc)])
## Other parameters
num.folds = 10 # number of folds
num.strata = 5 # number of strata
num.boot = 1000 # bootstrap replications
##------------------------------------------------------------------------------
## Empirical and Regression-adjusted DTE and PTE
##------------------------------------------------------------------------------
## DTE estimation
start.time = Sys.time()
dat_with_cdf_predictions = cdf.ML.adj(df, vec.loc, covariate_names, num.strata, num.folds, "xgboost")
res.dte = treatment.effect.estimation(dat_with_cdf_predictions, vec.loc, num.boot)
finish.time = Sys.time()
print(finish.time - start.time)
## PTE estimation
start.time = Sys.time()
dat_with_pdf_predictions = pdf.ML.adj(df, vec.loc.up, vec.loc.low, covariate_names, num.strata, num.folds, "xgboost")
res.pte = treatment.effect.estimation(dat_with_pdf_predictions, vec.loc.up, num.boot)
finish.time = Sys.time()
print(finish.time - start.time)
##------------------------------------------------------------------------------
## Plots
##------------------------------------------------------------------------------
## Set up the theme for plots
custom_theme = theme_bw() +
  theme(text=element_text(size=17)) +
  theme(axis.text.x = element_text(hjust=0.5),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 16, face = "bold"),
        plot.title = element_text(size = 16, face = "bold"),
        legend.text = element_text(size = 14, face = "bold"),
        legend.title = element_text(size = 16, face = "bold"),
        strip.text = element_text(size = 14, face = "bold"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

## DTE (empirical)
y.dte.max = max(max(res.dte$simple$est +1.96*res.dte$simple$boot.sd), max(res.dte$adjusted$est +1.96*res.dte$adjusted$boot.sd)) + 1e-5
y.dte.min = min(min(res.dte$simple$est -1.96*res.dte$simple$boot.sd), min(res.dte$adjusted$est -1.96*res.dte$adjusted$boot.sd)) - 1e-5

ggplot() +
  geom_line( aes(vec.loc, res.dte$simple$est - 1.96*res.dte$simple$boot.sd), color= cb_colors[1], linetype=2) +
  geom_line( aes(vec.loc, res.dte$simple$est + 1.96*res.dte$simple$boot.sd), color= cb_colors[1], linetype=2) +
  geom_ribbon(aes(x    = vec.loc,
                  ymin = res.dte$simple$est - 1.96*res.dte$simple$boot.sd,
                  ymax = res.dte$simple$est + 1.96*res.dte$simple$boot.sd),
              fill = cb_colors[1], alpha = .3) +
  geom_line( aes(vec.loc, res.dte$simple$est), color = cb_colors[1]) +
  ylim(y.dte.min, y.dte.max) +
  scale_x_continuous(breaks = seq(vec.loc.min, vec.loc.max, by= 2*bin.width), limit=c(vec.loc.min-0.5,vec.loc.max +0.5)) +
  geom_hline(yintercept=0, color="black", size=0.5, alpha = .3) +
  labs(title = "Empirical DTE",
       x= outcome_title, y="Probability") +
  custom_theme 
ggsave(paste0(outcome_name, "DTE_simple.boot.png"), width=5, height =3)

## DTE (adjusted)
ggplot() +
  geom_line( aes(vec.loc, res.dte$adjusted$est - 1.96*res.dte$adjusted$boot.sd), color= cb_colors[2], linetype=2) +
  geom_line( aes(vec.loc, res.dte$adjusted$est + 1.96*res.dte$adjusted$boot.sd), color= cb_colors[2], linetype=2) +
  geom_ribbon(aes(x    = vec.loc,
                  ymin = res.dte$adjusted$est - 1.96*res.dte$adjusted$boot.sd,
                  ymax = res.dte$adjusted$est + 1.96*res.dte$adjusted$boot.sd),
              fill = cb_colors[2], alpha = .3) +
  geom_line( aes(vec.loc, res.dte$adjusted$est), color = cb_colors[2]) +
  ylim(y.dte.min, y.dte.max) +
  scale_x_continuous(breaks = seq(vec.loc.min, vec.loc.max, by= 2*bin.width), limit=c(vec.loc.min-0.5,vec.loc.max +0.5)) +
  geom_hline(yintercept=0, color="black", size=0.5, alpha = 0.3) +
  labs(title = "Adjusted DTE",
       x= outcome_title, y="Probability") +
  custom_theme
ggsave(paste0(outcome_name, "DTE_adj.boot.png"), width=5, height =3)

## PTE (empirical)
y.max = max(max(res.pte$simple$est +1.96*res.pte$simple$boot.sd), max(res.pte$adjusted$est +1.96*res.pte$adjusted$boot.sd)) + 1e-5
y.min = min(min(res.pte$simple$est -1.96*res.pte$simple$boot.sd), min(res.pte$adjusted$est -1.96*res.pte$adjusted$boot.sd)) - 1e-5

ggplot(res.pte$simple, aes(vec.loc, est) ) +
   geom_bar( stat = "identity", color= cb_colors[1], fill=cb_colors[1]) +
   geom_errorbar(aes(ymin = est-1.96*boot.sd,
                     ymax = est+1.96*boot.sd),
                 #position=position_dodge(.9)
   ) +
   ylim(y.min, y.max) +
   geom_hline(yintercept=0, color="black", size=0.5, alpha = 0.3) +
   labs(title = "Empirical PTE", x= outcome_title, y="Probability")  +
   scale_x_continuous(breaks = seq(vec.loc.min, vec.loc.max, by= 2*bin.width), limit=c(vec.loc.min-bin.width,vec.loc.max + bin.width)) +
   custom_theme 
ggsave(paste0(outcome_name, "PTE_simple.boot.png"), width=5, height =3) 

## PTE (adjusted)
ggplot(res.pte$adjusted, aes(vec.loc, est) ) +
  geom_bar( stat = "identity", color= cb_colors[2], fill=cb_colors[2]) +
  geom_errorbar(aes(ymin = est-1.96*boot.sd,
                    ymax = est+1.96*boot.sd),
                                    # Width of the error bars
                #position=position_dodge(.9)
  ) +
  ylim(y.min, y.max) +
  geom_hline(yintercept=0, color="black", size=0.5, alpha = .3) +
  labs(title="Adjusted PTE", x= outcome_title, y="Probability")  +
  scale_x_continuous(breaks = seq(vec.loc.min, vec.loc.max, by= 2*bin.width), limit=c(vec.loc.min-bin.width,vec.loc.max + bin.width)) +
  custom_theme
ggsave(paste0(outcome_name, "PTE_adj.boot.png"), width=5, height =3) 

## Display SE reduction in % for DTE and PTE
dte.boot.sd.ratio = 100*(1-res.dte$adjusted$boot.sd/res.dte$simple$boot.sd)
print(dte.boot.sd.ratio)
pte.boot.sd.ratio = 100*(1-res.pte$adjusted$boot.sd/res.pte$simple$boot.sd)
print(pte.boot.sd.ratio)
