library(data.table)
library(xgboost)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(randomForest)
library(ggplot2)
library("Matrix")
library(pROC)
library(Rcpp)
setwd("C:/Users/mazhecheng/Desktop/")

sep_train_dier <- read.delim("C:/Users/mazhecheng/Desktop/sep_train_dier.csv")


df_train <- sep_train_dier %>% select(-c(key,dt))

#write.csv(df_train,file = "df_train.csv")

#设置训练集和测试
#sample(1:nrow(df_train))

train <- sample(nrow(df_train), 0.7*nrow(df_train))
df.train <- df_train[train,]
df.validate <- df_train[-train,]

#xgb矩阵�?


train_matrix <- sparse.model.matrix(risk_flag ~ .-1, data = df.train)
test_matrix <- sparse.model.matrix(risk_flag ~ .-1, data = df.validate)
train_label <- as.numeric(df.train$risk_flag)
test_label <-  as.numeric(df.validate$risk_flag)
train_fin <- list(data=train_matrix,label=train_label) 
test_fin <- list(data=test_matrix,label=test_label) 
dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)


#dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
#dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)

best_param = list()
best_seednumber = 12345
best_logloss = Inf
best_logloss_index = 0

# 自定义调参组�?
for (iter in 1:50) {
  param <- list(objective = "binary:logistic",     # 目标函数：logistic的二分类模型，因为Y值是二元�?
                eval_metric = c("logloss"),                # 评估指标：logloss
                max_depth = sample(6:10, 1),               # 最大深度的调节范围�?1�? 6-10 区间的数
                eta = runif(1, .01, 1.),                   # eta收缩步长调节范围�?1�? 0.01-1.0区间的数
                gamma = runif(1, 0.0, 0.2),                # gamma最小损失调节范围：1�? 0-0.2区间的数
                subsample = runif(1, .6, .9),             
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 50                                   # 迭代次数�?50
  cv.nfold = 5                                     # 5折交叉验�?
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, metrics=c("auc","rmse","error"),
                 nfold=cv.nfold, nrounds=cv.nround, watchlist = list(),
                 verbose = F, early_stop_round=8, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[,test_logloss_mean])
  min_logloss_index = which.min(mdcv$evaluation_log[,test_logloss_mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}




xgb_plot <- function(input,output){
  history=input
  train_history=history[,1:8]%>%mutate(id=row.names(history),class="train")
  test_history=history[,9:16]%>%mutate(id=row.names(history),class="test")
  colnames(train_history)=c("logloss.mean","logloss.std","auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
  colnames(test_history)=c("logloss.mean","logloss.std","auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
  
  his=rbind(train_history,test_history)
  his$id=his$id%>%as.numeric
  his$class=his$class%>%factor
  
  if(output=="auc"){ 
    auc=ggplot(data=his,aes(x=id, y=auc.mean,ymin=auc.mean-auc.std,ymax=auc.mean+auc.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation AUC")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(auc)
  }
  
  
  if(output=="rmse"){
    rmse=ggplot(data=his,aes(x=id, y=rmse.mean,ymin=rmse.mean-rmse.std,ymax=rmse.mean+rmse.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation RMSE")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(rmse)
  }
  
  if(output=="error"){
    error=ggplot(data=his,aes(x=id,y=error.mean,ymin=error.mean-error.std,ymax=error.mean+error.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation ERROR")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(error)
  }
  
}


xgb_plot(mdcv$evaluation_log[,-1]%>%data.frame,"auc")

xgb_plot(mdcv$evaluation_log[,-1]%>%data.frame,"rmse")

xgb_plot(mdcv$evaluation_log[,-1]%>%data.frame,"error")


print(best_param)
print(best_logloss_index)

zc.xgb.a <- xgb.train(data=dtrain, params=best_param, nrounds=50, nthread=6, watchlist = list())

#zc_xgb <- xgb.train(data=dtrain, params=best_param, nrounds=, nthread=6, watchlist = list())

importanceRaw <- xgb.importance(feature_names=colnames(dtrain), model = zc.xgb.a)

head(importanceRaw,n=10)

xgb.plot.importance(importanceRaw) 

pre_xgb = round(predict(zc.xgb.a,newdata = dtest))

table(test_label,pre_xgb,dnn=c("true","pre"))

xgboost_roc <- roc(test_label,as.numeric(pre_xgb))
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", 
     print.thres=TRUE,main='ROC curve')
xgb.save(zc.xgb.a, 'zc_xgb.model')

##---------------------importance方案-----------------##
cum_impt <- data.frame(names = importanceRaw$Feature,impt= cumsum(importanceRaw$Importance))
cum_impt <- filter(cum_impt,cum_impt$impt<0.9)
selected_feature<-cum_impt$names

impot_data <- df %>% select(selected_feature,risk_flag)

#设置训练集和测试
sample(1:nrow(impot_data))

impot_train <- sample(nrow(impot_data), 0.7*nrow(impot_data))
impdf.train <- impot_data[impot_train,]
impdf.validate <- impot_data[-impot_train,]

#xgb矩阵�?

imp_train_matrix <- sparse.model.matrix(risk_flag ~ .-1, data = impdf.train)
imp_test_matrix <- sparse.model.matrix(risk_flag ~ .-1, data = impdf.validate)

imp_train_label <- as.numeric(impdf.train$risk_flag)
imp_test_label <-  as.numeric(impdf.validate$risk_flag)

imp_train_fin <- list(data=imp_train_matrix,label=imp_train_label) 
imp_test_fin <- list(data=imp_test_matrix,label=imp_test_label) 

imp_dtrain <- xgb.DMatrix(data = imp_train_fin$data, label = imp_train_fin$label) 
imp_dtest <- xgb.DMatrix(data = imp_test_fin$data, label = imp_test_fin$label)

xgb_2 <- xgb.train(data=imp_dtrain, params=best_param, nrounds=36, nthread=6, watchlist = list())

importanceR2 <- xgb.importance(feature_names=colnames(imp_dtrain), model = xgb_2)

head(importanceR2,n=10)

xgb.plot.importance(importanceR2) 

pre_m2 = round(predict(xgb_2,newdata = imp_dtest))

table(imp_test_label,pre_m2,dnn=c("true","pre"))

xgb2_roc <- roc(imp_test_label,as.numeric(pre_m2))
plot(xgb2_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", 
     print.thres=TRUE,main='ROC curve')

xgb.save(zc.xgb.a, 'zc_xgb.model')
# #--------------------------------------------------------------------------------------
# # feature selection    # 这里可以根据importance设置阈值，进行特征筛选，这是特征筛选的方式之一
# cum_impt=data.frame(names=importanceRaw$Feature,impt=cumsum(importanceRaw$Importance))
# cum_impt=filter(cum_impt,cum_impt$impt<0.9)
# selected_feature<-cum_impt$names
# 
# train=select(train,selected_feature)
# dtrain<- xgb.DMatrix(data=select(train,-bad)%>%as.matrix,label= train$bad%>%as.matrix)
# 
# model <- xgb.train(data=dtrain, params=best_param, nrounds=nround, nthread=6, watchlist = list())
# #----

library("plyr")
library("r2pmml")
library("pmml")

library(xgboost)
xgb.save(zc_xgb, 'zc_xgb.model')

print(best_logloss_index)

regist_model_v0.2_20200914
regist_model_v0.1