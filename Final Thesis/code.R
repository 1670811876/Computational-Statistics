library(ggplot2)
library(fBasics)

setwd("E:\\Courses\\Computational Statistics\\Final Thesis")

data <- read.csv("pufa_stock.csv", encoding="utf-8")

colnames(data)[1] <- "date"
colnames(data)

data <- data[order(data$date),]
rownames(data) <- c(1:nrow(data))

data$date <- as.character(data$date)

data.train <- data[data$date <= "2019-05-28",]
data.test <- data[data$date > "2019-05-28",]
rownames(data.test) <- c(1:nrow(data.test))

data$date <- as.Date(data$date, "%Y-%m-%d")

ggplot() + geom_line(data, mapping=aes(x=date, y=close, group=factor(1)), col="blue") + 
  xlab('Year') + 
  theme(axis.text.x=element_text(angle=45, hjust=0.5, vjust=0.5)) +
  theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
        panel.background=element_blank(), axis.line=element_line(colour="black")) + 
  scale_x_date(date_breaks='100 days', date_labels='%Y-%m')

data$log_close <- log(data$close)
return_data <- data.frame(date=data[['date']][2:nrow(data)],
                          return=diff(data$log_close))

basicStats(data$close)
basicStats(return_data$return)

ggplot(data, aes(x=close)) + geom_density(colour="steelblue", lwd=2) + 
  theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
        panel.background=element_blank(), axis.line=element_line(colour="black"))
ggplot(return_data, aes(x=return)) + geom_density(colour="steelblue", lwd=2) + 
  theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
        panel.background=element_blank(), axis.line=element_line(colour="black"))

# 历史模拟法计算VaR值
dd <- diff(log(data.train$close))
loss <- -dd * 100
VaR <- quantile(loss, 0.95)
VaR

# 蒙特卡洛模拟计算VaR值
library(tseries)
adf.test(data$close, alternative="stationary")
acf(data$close)
pacf(data$close)

## 建立GARCH模型
library(rugarch)
spec <- ugarchspec(mean.model=list(armaOrder=c(0,0)), variance.model=list(garchOrder=c(1,1)),
                   distribution.model='std')
fit <- ugarchfit(data=loss, spec=spec)
result <- coef(fit)

mu <- result[['mu']]
alpha <- c(result[['omega']], result[['alpha1']])
beta <- result[['beta1']]
df <- result[['shape']]
sig <- sigma(fit)

## 设置天数为一周，也就是5天
t <- 5
## 迭代次数
nround <- 5000

library(fGarch)
set.seed(123)
err <- matrix(rstd(t*nround, mean=0, sd=1, nu=df), t, nround)
init <- c(loss[length(loss)], sig[length(loss)])
xt <- NULL

# 以init为起点，进行nround轮迭代
for (j in 1:nround){
  lt <- NULL  # 初始化为空值
  at <- init[1] - mu  # 初始化残差
  vart <- init[2]^2  # 初始化方差
  for (i in 1:t){
    var <- alpha[1] + alpha[2]*at[i]^2 + beta*vart[i]  # 根据GARCH模型拟合出下一期方差
    vart <- c(vart,var)  # 前i期方差
    at <- c(at,sqrt(var)*err[i,j])  # 前i期残差
    lt <- c(lt, mu+at[i+1])  # 前i期的损失变量
  }  # 此循环结束后，得到未来5期的损失变量序列的一次模拟值lt
  xt <- c(xt,sum(lt))  # 未来5期的损失变量的一次总和
}  # 此循环结束后就得到5期损失变量总和的3000次模拟值

VaR2 <- quantile(xt, 0.95)
VaR2


# 进行Kupiec检验
num_his <- 0
num_mc <- 0 
for (r in 1:nrow(data.test)) {
  if (r == 1) {
    data_his <- data.train
  } else {
    data_his <- rbind(data.train, data.test[1:(r-1),])
  }
  dd <- diff(log(data_his$close))
  loss <- -dd * 100
  VaR <- quantile(loss, 0.95)
  
  spec <- ugarchspec(mean.model=list(armaOrder=c(0,0)), variance.model=list(garchOrder=c(1,1)),
                     distribution.model='std')
  fit <- ugarchfit(data=loss, spec=spec)
  result <- coef(fit)
  
  mu <- result[['mu']]
  alpha <- c(result[['omega']], result[['alpha1']])
  beta <- result[['beta1']]
  df <- result[['shape']]
  sig <- sigma(fit)
  t <- 5
  nround <- 7000
  
  err <- matrix(rstd(t*nround, mean=0, sd=1, nu=df), t, nround)
  init <- c(loss[length(loss)], sig[length(loss)])
  xt <- NULL
  
  for (j in 1:nround){
    lt <- NULL
    at <- init[1] - mu
    vart <- init[2]^2 
    for (i in 1:t){
      var <- alpha[1] + alpha[2]*at[i]^2 + beta*vart[i] 
      vart <- c(vart,var)  
      at <- c(at,sqrt(var)*err[i,j])
      lt <- c(lt, mu+at[i+1]) 
    } 
    xt <- c(xt,sum(lt)) 
  } 
  VaR2 <- quantile(xt, 0.95)
  
  if (r == 1) {
    ac_loss <- data.train[['close']][nrow(data.train)] - data.test[['close']][1]
  } else {
    ac_loss <- data.test[['close']][r-1] - data.test[['close']][r]
  }
  
  if (ac_loss > VaR[[1]] * 0.01) {
    num_his <- num_his + 1
  } 
  
  if (ac_loss > VaR2[[1]] * 0.01) {
    num_mc <- num_mc + 1
  } 
}

print(num_his)
print(num_mc)

VaR.back.test <- function(T, N) {
  p <- 0.05
  con.level <- 0.05
  prob <- 1 - p
  LR_POF <- -2*log((prob^N)*(1-prob)^(T-N))+2*log(((N/T)^N)*(1-N/T)^(T-N))
  critical <- qchisq(con.level, df=1);
  P_value <- pchisq(LR_POF, df=1, lower.tail=F)
  list(LR_POF,P_value)
}

VaR.back.test(nrow(data.test), num_his)
VaR.back.test(nrow(data.test), 50)
