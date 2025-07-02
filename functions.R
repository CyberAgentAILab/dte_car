cdf.ML.adj = function(dat, vec.loc, covariate_names, num.strata, num.folds, model){
  ###
  # Input: dat is the data frame with the outcome and covariates
  #        vec.loc is the vector of locations for DTE
  #        covariate_names is the vector of covariate names
  #        num.strata is the number of strata
  #        num.folds is the number of folds for cross-fitting
  #        model is the model to be used for estimation (ols, xgboost, etc.)
  # Output: data frame with the outcome, covariates and the predictions of the outcome at each location
  ###
  # number of locations for DTE
  n.loc = length(vec.loc)
  # Create folds for cross-fitting
  dat$folds = sample(c(1:num.folds), nrow(dat), replace=TRUE) 
  # Compute outcome below level y 
  mat.d.y = 1 * outer(dat$Y, vec.loc, "<=")  ## n x n.loc
  if (model=="ols"){
    for(j in 1:n.loc){
      dat[,paste('Y_', j, sep = '')] = mat.d.y[,j]
      for (d in 1:2){
        d.treat = 2 - d # 1 is treatment (1), 2 is control (0)
        for (s in 1:num.strata){
          dat[,paste('Y_', j, '_hat_d', d.treat, sep = '')] = 0
          for (f in 1:num.folds){
            # Fit OLS model
            model = lm(as.formula(paste('Y', '_', j, '~', paste(covariate_names, collapse='+'), sep = '')), 
                       data=dat %>% filter(folds != f, S == s, W==d.treat))
            # Predict the outcome
            dat[(dat$folds==f & dat$S==s), paste('Y_', j, '_hat_d', d.treat, sep = '')] = predict(model, newdata=dat %>% filter(folds == f, S==s))
          }
        }
      }
    }
  }
  else if (model == 'xgboost'){
    for(j in 1:n.loc){
      dat[,paste('Y_', j, sep = '')] = mat.d.y[,j]
      for (d in 1:2){
        d.treat = 2 - d # 1 is treatment (1), 2 is control (0)
        for (s in 1:num.strata){
          dat[,paste('Y_', j, '_hat_d', d.treat, sep = '')] = 0
          for (f in 1:num.folds){
            # Fit xgboost model
            model = xgboost(data = dat %>% filter(folds != f, S == s, W==d.treat) %>% select(all_of(covariate_names)) %>% as.matrix(), 
                            label = dat %>% filter(folds != f, S == s, W==d.treat) %>% select(paste('Y_', j, sep = '')) %>% unlist() %>% as.numeric(),
                            max_depth = 3, eta = 0.1, nrounds = 300, objective = "binary:logistic", verbose = 0)
            # Predict the outcome
            dat[(dat$folds==f & dat$S==s), paste('Y_', j, '_hat_d', d.treat, sep = '')]  = 
              predict(model, newdata=dat %>% filter(folds == f, S==s) %>% select(all_of(covariate_names)) %>% as.matrix())
          }
        }
      }
    }
  }
  return (dat)
}
pdf.ML.adj = function(dat, vec.loc.up, vec.loc.low, covariate_names, num.strata, num.folds, model){
  ###
  # Input: dat is the data frame with the outcome and covariates
  #        vec.loc.up is the vector of locations for PTE (upper bounds)
  #        vec.loc.low is the vector of locations for PTE (lower bounds)
  #        covariate_names is the vector of covariate names
  #        num.strata is the number of strata
  #        num.folds is the number of folds for cross-fitting
  #        model is the model to be used for estimation (ols, xgboost, etc.)
  # Output: data frame with the outcome, covariates and the predictions of the outcome at each location
  ###
  # number of locations for PTE
  n.loc = length(vec.loc.up)
  # Create folds for cross-fitting
  dat$folds = sample(c(1:num.folds), nrow(dat), replace=TRUE) 
  # Compute outcome below level y 
  mat.d.y.up = 1 * outer(dat$Y, vec.loc.up, "<=")  ## n x n.loc
  mat.d.y.low = 1 * outer(dat$Y, vec.loc.low, ">")  ## n x n.loc
  mat.d.y = mat.d.y.up*mat.d.y.low
  if (model=="ols"){
    for(j in 1:n.loc){
      dat[,paste('Y_', j, sep = '')] = mat.d.y[,j]
      for (d in 1:2){
        d.treat = 2 - d # 1 is treatment (1), 2 is control (0)
        for (s in 1:num.strata){
          dat[,paste('Y_', j, '_hat_d', d.treat, sep = '')] = 0
          for (f in 1:num.folds){
            # Fit OLS model
            model = lm(as.formula(paste('Y', '_', j, '~', paste(covariate_names, collapse='+'), sep = '')), 
                       data=dat %>% filter(folds != f, S == s, W==d.treat))
            # Predict the outcome
            dat[(dat$folds==f & dat$S==s), paste('Y_', j, '_hat_d', d.treat, sep = '')] = predict(model, newdata=dat %>% filter(folds == f, S==s))
          }
        }
      }
    }
  }
  else if (model == 'xgboost'){
    for(j in 1:n.loc){
      dat[,paste('Y_', j, sep = '')] = mat.d.y[,j]
      for (d in 1:2){
        d.treat = 2 - d # 1 is treatment (1), 2 is control (0)
        for (s in 1:num.strata){
          dat[,paste('Y_', j, '_hat_d', d.treat, sep = '')] = 0
          for (f in 1:num.folds){
            # Fit xgboost model
            model = xgboost(data = dat %>% filter(folds != f, S == s, W==d.treat) %>% select(all_of(covariate_names)) %>% as.matrix(), 
                            label = dat %>% filter(folds != f, S == s, W==d.treat) %>% select(paste('Y_', j, sep = '')) %>% unlist() %>% as.numeric(),
                            max_depth = 3, eta = 0.1, nrounds = 300, objective = "binary:logistic", verbose = 0)
            # Predict the outcome
            dat[(dat$folds==f & dat$S==s), paste('Y_', j, '_hat_d', d.treat, sep = '')]  = 
              predict(model, newdata=dat %>% filter(folds == f, S==s) %>% select(all_of(covariate_names)) %>% as.matrix())
          }
        }
      }
    }
  }
  return (dat)
}
treatment.effect.estimation = function(dat_with_predictions, vec.loc, B.size){
    ###
    # Input: dat_with_predictions is the data frame with the outcome, covariates and the predictions of the outcome at each location
    #        vec.loc is the vector of locations
    # Output: simple and regression-adjusted DTE or PTE estimates and standard errors
    ###
    # number of locations
    n.loc = length(vec.loc)
    # Initialize vectors
    vec.dte = rep(NA, n.loc)
    vec.dte.sd = rep(NA, n.loc)
    vec.dte.boot.sd = rep(NA, n.loc)
    vec.dte.ra = rep(NA, n.loc)
    vec.dte.ra.sd = rep(NA, n.loc)
    vec.dte.ra.moment.sd = rep(NA, n.loc)
    vec.dte.ra.boot.sd = rep(NA, n.loc)
    # Compute DTE or PTE estimates
    for (j in 1:n.loc){
      # Empirical point estimate and standard error
      temp = dat_with_predictions %>% mutate(cdf.1 = !!sym(paste('Y_', j, sep=''))*W/prob_treatment,
                                             cdf.0 = !!sym(paste('Y_', j, sep=''))*(1-W)/(1-prob_treatment),
                                             term.1.tilde = !!sym(paste('Y_', j, sep=''))/prob_treatment, 
                                             term.0.tilde = -!!sym(paste('Y_', j, sep=''))/(1-prob_treatment)
      ) %>% group_by(S) %>%
        mutate(
          term.1.hat = term.1.tilde - sum(term.1.tilde*(W == 1), na.rm = TRUE)/sum(W==1, na.rm = TRUE),
          term.0.hat = term.0.tilde - sum(term.0.tilde*(W == 0), na.rm = TRUE)/sum(W==0, na.rm = TRUE),
          term.2.1 = sum(!!sym(paste('Y_', j, sep=''))*(W==1), na.rm = TRUE)/sum(W==1, na.rm = TRUE),
          term.2.0 = sum(!!sym(paste('Y_', j, sep=''))*(W==0), na.rm = TRUE)/sum(W==0, na.rm = TRUE),
          dte.omega = W*term.1.hat^2 + (1-W)*term.0.hat^2 + (term.2.1 - term.2.0)^2
        ) %>%
        ungroup()
      # point estimate
      temp$dte = temp$cdf.1 - temp$cdf.0
      vec.dte[j] = mean(temp$dte)
      # standard error 
      vec.dte.sd[j] = sqrt(mean(temp$dte.omega))/sqrt(nrow(temp))
      # influence function
      temp$influence.function = temp$dte - mean(temp$dte)
      # multiplier bootstrap
      boot.draw = rep(0, length = B.size)
      for (b in 1:B.size){
        # Mammen multiplier
        eta1 = rnorm(nrow(temp), 0, 1)
        eta2 = rnorm(nrow(temp), 0, 1)
        xi = eta1/sqrt(2) + (eta2^2-1)/2
        boot.draw[b] = mean(temp$dte) +mean(xi*temp$influence.function)
      }
      vec.dte.boot.sd[j] = sd(boot.draw)
      # Regression-adjusted point estimate and standard error
      temp.ra = dat_with_predictions %>% mutate(cdf.1.ra = (!!sym(paste('Y_', j, sep=''))-!!sym(paste('Y_', j, '_hat_d1', sep='')))*W/prob_treatment +
                                                  !!sym(paste('Y_', j, '_hat_d1', sep='')),
                                                cdf.0.ra = (!!sym(paste('Y_', j, sep=''))-!!sym(paste('Y_', j, '_hat_d0', sep='')))*(1-W)/(1-prob_treatment)+
                                                  !!sym(paste('Y_', j, '_hat_d0',sep='')),
                                                term.1.tilde = (1-1/prob_treatment)*!!sym(paste('Y_', j, '_hat_d1', sep='')) - !!sym(paste('Y_', j, '_hat_d0', sep='')) +
                                                  !!sym(paste('Y_', j, sep=''))/prob_treatment, 
                                                term.0.tilde = (1/(1-prob_treatment)-1)*!!sym(paste('Y_', j, '_hat_d0', sep='')) + !!sym(paste('Y_', j, '_hat_d1', sep=''))- 
                                                  !!sym(paste('Y_', j, sep=''))/(1-prob_treatment)) %>%
        group_by(S) %>%
        mutate(
          term.1.hat = term.1.tilde - sum(term.1.tilde*(W == 1), na.rm = TRUE)/sum(W==1, na.rm = TRUE),
          term.0.hat = term.0.tilde - sum(term.0.tilde*(W == 0), na.rm = TRUE)/sum(W==0, na.rm = TRUE),
          term.2.1 = sum(!!sym(paste('Y_', j, sep=''))*(W==1), na.rm = TRUE)/sum(W==1, na.rm = TRUE),
          term.2.0 = sum(!!sym(paste('Y_', j, sep=''))*(W==0), na.rm = TRUE)/sum(W==0, na.rm = TRUE),
          dte.omega = W*term.1.hat^2 + (1-W)*term.0.hat^2 + (term.2.1 - term.2.0)^2
        ) %>%
        ungroup()
      # point estimate
      temp.ra$dte.ra = temp.ra$cdf.1.ra - temp.ra$cdf.0.ra
      vec.dte.ra[j] = mean(temp.ra$dte.ra)
      # standard error
      vec.dte.ra.sd[j] = sqrt(mean(temp.ra$dte.omega))/sqrt(nrow(temp.ra))
      # influence function
      temp.ra$influence.function = temp.ra$dte.ra - mean(temp.ra$dte.ra)
      # multiplier bootstrap
      boot.ra.draw = rep(0, length = B.size)
      for (b in 1:B.size){
        # Mammen multiplier
        eta1 = rnorm(nrow(temp), 0, 1)
        eta2 = rnorm(nrow(temp), 0, 1)
        xi = eta1/sqrt(2) + (eta2^2-1)/2
        boot.ra.draw[b] = mean(temp.ra$dte.ra) + mean(xi*temp.ra$influence.function)
      }
      vec.dte.ra.boot.sd[j] = sd(boot.ra.draw)
    }
    est.simple    = data.frame( cbind(vec.loc, vec.dte, vec.dte.sd, vec.dte.boot.sd) )
    est.adj = data.frame( cbind(vec.loc, vec.dte.ra, vec.dte.ra.sd, vec.dte.ra.boot.sd))
    colnames(est.simple) = c("vec.loc", "est", "sd", "boot.sd")
    colnames(est.adj) = c("vec.loc", "est", "sd", "boot.sd")
    return(list(simple = est.simple, adjusted = est.adj))
  }
