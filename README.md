# Oversampling with SMOTE with its relative algorithms

[![View TRIM-SBR: TRIM-Smoothed Bootstrap Resampling]
[(https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/75168-oversampling-imbalanced-data-smote-related-algorithms)

   -  SMOTE (Chawla, NV. et al. 2002)[1]  
   -  Borderline SMOTE (Han, H. et al. 2005)[2]  
   -  ADASYN (He, H. et al. 2008)[3]  
   -  Safe-level SMOTE (Bunkhumpornpat, C. at al. 2009)[4]  

TRIM-Smoothed Bootstrap Resampling (TRIM-SBR) is a proposed method to reduce the overgeneralization problem that usually occurs when synthetic data is formed into a majority class region with evenly distributed synthetic data effects. Our method is based on pruning by looking for a particular minority area while maintaining the generality of the data so that it will find the minority data set while filtering out irrelevant data. The pruning results will produce a minority data seed that is used as a benchmark in duplicating data. To ensure the duplication of data is evenly distributed, the bootstrap resampling technique is used to create new data

# Check the number of data for each class
```matlab
label0 = repmat("class0",length(data),1);
label1 = repmat("class1",length(data1),1);
label2 = repmat("class2",length(data2),1);

dataset = array2table([data;data1;data2]);
dataset = addvars(dataset, [label0;label1;label2],...
    'NewVariableNames','label');
labels = dataset(:,end);
t = tabulate(dataset.label)
```
| |1|2|3|
|:--:|:--:|:--:|:--:|
|1|'class0'|1338|88.9037|
|2|'class1'|152|10.0997|
|3|'class2'|15|0.9967|

```matlab
uniqueLabels = string(t(:,1));
labelCounts = cell2mat(t(:,2));
```

![figure_1.png](README_images/figure_1.png)

![figure_2.png](README_images/figure_2.png)

# Reference and its graphical explanation


[1]: Chawla, N. V., Bowyer, K. W., Hall, L. O., \& Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357. 


[2]: Han, H., Wang, W. Y., \& Mao, B. H. (2005). Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning. In International conference on intelligent computing (pp. 878-887). Springer, Berlin, Heidelberg. 


[3]: He, H., Bai, Y., Garcia, E. A., \& Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. In 2008 IEEE International Joint Conference on Neural Networks (pp. 1322-1328). IEEE. 


[4]: Bunkhumpornpat, C., Sinapiromsaran, K., \& Lursinsap, C. (2009). Safe-level-smote: Safe-level-synthetic minority over-sampling technique for handling the class imbalanced problem. In Pacific-Asia conference on knowledge discovery and data mining (pp. 475-482). Springer, Berlin, Heidelberg.


## SMOTE (Chawla, NV. et al. 2002)


![image_0.png](README_images/image_0.png)

