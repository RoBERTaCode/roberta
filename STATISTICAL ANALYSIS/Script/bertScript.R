setwd(".") #set local dir here
library(exact2x2)
library(effsize)
library(xtable)



datasets=c("JAVA","ANDROID")
masklevels=c("TOKEN","CONSTRUCT","BLOCK")
abstractions=c("ABSTRACT","RAW")

###########################################################
#Comparison between abstract and raw - perfect prediction
###########################################################

res=list(Dataset=c(),Masking=c(),McNemar.p=c(),McNemar.OR=c())

for(d in datasets)
{
  for(l in masklevels)
  {
    print(paste(d,l))
    nameAbstract=paste("predictions/",d,"_",l,"_ABSTRACT/raw_data.csv",sep="")
    nameRaw=paste("predictions/",d,"_",l,"_RAW/raw_data.csv",sep="")
    t1<-read.csv(nameRaw,na.strings = "None")
    t2<-read.csv(nameAbstract,na.strings = "None")
    t3=merge.data.frame(t1,t2,by.x = c("method_id","masked_method_id","k"),by.y=c("method_id","masked_method_id","k"))
    attach(t3,warn.conflicts = FALSE)
    mn=mcnemar.exact(is_perfect_prediction.x,is_perfect_prediction.y)
    res$Dataset=c(res$Dataset,as.character(d))
    res$Masking=c(res$Masking,as.character(l))
    res$McNemar.p=c(res$McNemar.p,mn$p.value)
    res$McNemar.OR=c(res$McNemar.OR,mn$estimate)
  }
}

res=data.frame(res)
res2=res
#p-value adjustment
res2$McNemar.p=p.adjust(res2$McNemar.p,method="BH")
print(xtable(res2),include.rownames=FALSE)


###########################################################
#Comparison between abstract and raw - BLEU score and Levenshten distance
###########################################################

res=list(Dataset=c(),Masking=c(),Leven.p=c(),Leven.d=c(),BLEU1.p=c(),BLEU1.d=c(),BLEU2.p=c(),BLEU2.d=c(),BLEU3.p=c(),BLEU3.d=c(),BLEU4.p=c(),BLEU4.d=c())

for(d in datasets)
{
  for(l in masklevels)
  {
    print(paste(d,l))
    nameAbstract=paste("predictions/",d,"_",l,"_ABSTRACT/raw_data.csv",sep="")
    nameRaw=paste("predictions/",d,"_",l,"_RAW/raw_data.csv",sep="")
    t1<-read.csv(nameRaw,na.strings = "None")
    t2<-read.csv(nameAbstract,na.strings = "None")
    t3=merge.data.frame(t1,t2,by.x = c("method_id","masked_method_id","k"),by.y=c("method_id","masked_method_id","k"))
    attach(t3,warn.conflicts = FALSE)
    #mn=mcnemar.exact(is_perfect_prediction.x,is_perfect_prediction.y)
    w=wilcox.test(t3$lev_distance.x,t3$lev_distance.y,paired=TRUE)
    cliffd=cliff.delta(t3$lev_distance.x,t3$lev_distance.y,paired=TRUE)
    res$Leven.d=c(res$Leven.d,cliffd$estimate)
    res$Leven.p=c(res$Leven.p,w$p.value)
    res$Dataset=c(res$Dataset,as.character(d))
    res$Masking=c(res$Masking,as.character(l))
    for (b in 1:4)
    {
      d1=t3[[paste("bleu",b,".x",sep="")]]
      d2=t3[[paste("bleu",b,".y",sep="")]]
      w=wilcox.test(d1,d2,paired=TRUE)
      res[[paste("BLEU",b,".p",sep="")]]=c(res[[paste("BLEU",b,".p",sep="")]],w$p.value)
      cliffd=cliff.delta(d1,d2,paired=TRUE)
      res[[paste("BLEU",b,".d",sep="")]]=c(res[[paste("BLEU",b,".d",sep="")]],cliffd$estimate)
      #res[[paste("BLEU",b,".d",sep="")]]=c(res[[paste("BLEU",b,".d",sep="")]],NA)
    }
  }
}


res=data.frame(res)
res2=res
#adjustment
allp=c(res2$Leven.p,res2$BLEU1.p,res2$BLEU2.p,res2$BLEU3.p,res2$BLEU4.p)
adjp=p.adjust(allp,method="BH")
res2$Leven.p=adjp[1:6]
for(x in 1:4)
{
  lower=x*6+1
  upper=x*6+6
  res2[[paste("BLEU",x,".p",sep="")]]=adjp[lower:upper]
}
print(xtable(res2),include.rownames=FALSE)



###########################################################
## compare Java and Android - perfect prediction
###########################################################

res=list(Abstraction=c(),Masking=c(),Fisher.p=c(),Fisher.OR=c())

for(l in masklevels)
{
  for (a in abstractions)
  {
    print(paste(a,l))
    nameJava=paste("predictions/","JAVA","_",l,"_",a,"/raw_data.csv",sep="")
    nameAndroid=paste("predictions/","ANDROID","_",l,"_",a,"/raw_data.csv",sep="")
    t1<-read.csv(nameJava,na.strings = "None")
    t2<-read.csv(nameAndroid,na.strings = "None")
    m=array(c(sum(t1$is_perfect_prediction=="True"),
              sum(t1$is_perfect_prediction=="False"),
              sum(t2$is_perfect_prediction=="True"),
              sum(t2$is_perfect_prediction=="False")),dim=c(2,2))
    
    res$Abstraction=c(res$Abstraction,as.character(a))
    res$Masking=c(res$Masking,as.character(l))
    
    f=fisher.test(m)
    res$Fisher.p=c(res$Fisher.p,f$p.value)
    res$Fisher.OR=c(res$Fisher.OR,1/f$estimate)
    
  }
}
res=data.frame(res)
res2=res
#p-value adjustment
res2$Fisher.p=p.adjust(res2$Fisher.p,method="BH")

print(xtable(res2),include.rownames=FALSE)


###########################################################
## compare Java and Android - BLEU score and Levenshtein distance
###########################################################
res=list(Abstraction=c(),Masking=c(),Leven.p=c(),Leven.d=c(),BLEU1.p=c(),BLEU1.d=c(),BLEU2.p=c(),BLEU2.d=c(),BLEU3.p=c(),BLEU3.d=c(),BLEU4.p=c(),BLEU4.d=c())

for(l in masklevels)
{
  for (a in abstractions)
  {
    print(paste(a,l))
    nameJava=paste("predictions/","JAVA","_",l,"_",a,"/raw_data.csv",sep="")
    nameAndroid=paste("predictions/","ANDROID","_",l,"_",a,"/raw_data.csv",sep="")
    t1<-read.csv(nameJava,na.strings = "None")
    t2<-read.csv(nameAndroid,na.strings = "None")
    m=array(c(sum(t1$is_perfect_prediction=="True"),
              sum(t1$is_perfect_prediction=="False"),
              sum(t2$is_perfect_prediction=="True"),
              sum(t2$is_perfect_prediction=="False")),dim=c(2,2))
    
    #t3=merge.data.frame(t1,t2,by.x = c("method_id","masked_method_id","k"),by.y=c("method_id","masked_method_id","k"))
    res$Abstraction=c(res$Abstraction,as.character(a))
    res$Masking=c(res$Masking,as.character(l))
    w=wilcox.test(t1$lev_distance,t2$lev_distance,paired=FALSE)
    res$Leven.p=c(res$Leven.p,w$p.value)
    cliffd=cliff.delta(t1$lev_distance,t2$lev_distance,paired=FALSE)
    res$Leven.d=c(res$Leven.d,cliffd$estimate)
    for (b in 1:4)
    {
      d1=t1[[paste("bleu",b,"",sep="")]]
      d2=t2[[paste("bleu",b,"",sep="")]]
      w=wilcox.test(d1,d2,paired=FALSE)
      res[[paste("BLEU",b,".p",sep="")]]=c(res[[paste("BLEU",b,".p",sep="")]],w$p.value)
      cliffd=cliff.delta(d1,d2,paired=FALSE)
      res[[paste("BLEU",b,".d",sep="")]]=c(res[[paste("BLEU",b,".d",sep="")]],cliffd$estimate)
      #res[[paste("BLEU",b,".d",sep="")]]=c(res[[paste("BLEU",b,".d",sep="")]],NA)
    }
    
  }
}
res=data.frame(res)
res2=res
#p-value adjustment
allp=c(res2$Leven.p,res2$BLEU1.p,res2$BLEU2.p,res2$BLEU3.p,res2$BLEU4.p)
adjp=p.adjust(allp,method="BH")
res2$Leven.p=adjp[1:6]
for(x in 1:4)
{
  lower=x*6+1
  upper=x*6+6
  res2[[paste("BLEU",x,".p",sep="")]]=adjp[lower:upper]
}


print(xtable(res2),include.rownames=FALSE)


############################################################
## N-GRAM VS. ROBERTA COMPARISON
############################################################

res=list(Dataset=c(),Masking=c(),Abstraction=c(),McNemar.p=c(),McNemar.OR=c())
for(d in datasets){
  for(l in masklevels){
    for(a in abstractions){
      t<-read.csv(paste("BASELINE_VS_ROBERTA_SHARED/",d,"_",l,"_",a,"/result_comparison.csv",sep=""))
      m=mcnemar.exact(t$is_perfect_prediction.baseline,t$is_perfect_prediction.RoBERTa)
      res$Dataset=c(res$Dataset,as.character(d))
      res$Masking=c(res$Masking,as.character(l))
      res$Abstraction=c(res$Abstraction,as.character(a))
      res$McNemar.p=c(res$McNemar.p, m$p.value)
      res$McNemar.OR=c(res$McNemar.OR,m$estimate)
    }
  }
}

res=data.frame(res)
#p-value adjustment
res$McNemar.p=p.adjust(res$McNemar.p,method="BH")
print(xtable(res),include.rownames=FALSE)
