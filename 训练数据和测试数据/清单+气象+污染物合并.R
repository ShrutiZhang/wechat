library(plyr)
###历史逐月训练数据: 气象+排放+污染物
mydata <- read.csv("D:\\machine\\data\\ozone\\metadjustment\\metozone1320_new_ssr_2021.csv")
mydata_monthmean <- aggregate(mydata,by=list(lon=mydata$lon,lat=mydata$lat,month=substr(mydata$date,1,6)),FUN="mean")
ozone <- mydata_monthmean[c(1,2,3,8)]
met_his <- read.csv('D:/machine/data/NCAR/era5/totalmet_history_2013-2020.csv')
met_his$month <- substr(gsub('-','',met_his$time),1,6)
emission <- read.csv('D:/machine/data/meic/meic1317/1317_10_species.csv')
emission$month <- as.character(emission$month)
dataall_his1 <- merge(ozone,met_his,by=c("lon","lat","month"))
dataall_his2 <- merge(dataall_his1,emission,by=c("lon","lat","month"))[c(-5)]
colnames(dataall_his2)###得到是13-17年历史数据
##
###ssp预测数据准备
sspmet_wd <- 'D:/machine/data/NCAR/ssp_ncsub_prd/'
sspmet <- list.files(sspmet_wd,pattern = ".csv")

