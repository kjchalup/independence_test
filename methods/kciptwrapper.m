function pval = kciptwrapper(X, Y, Z)
  options.null_estimate = 'bootstrap';
  options.distance = 'symmetric_regression'; %'rkhs';
  options.bootstrap_samples = 10000;
  boptions.bootstrap_samples = 25;
  
  kX = rbf(median_pdist(X));
  kY = rbf(median_pdist(Y));
  kZ = rbf(median_pdist(Z));

  new_test = bootstrap(@kcipt, boptions);
  [statistic null] = new_test(X, Y, Z, kX, kY, kZ, options);
  pval = null.pvalue(statistic);
