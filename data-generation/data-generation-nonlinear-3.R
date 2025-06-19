p=50
T=100

set.seed(313)

model1=4
for(model2 in c(3)){
  for(n.train in 1000*(1:8)){
    dir.create(paste("./data/model",model1,"-",model2,"-",n.train,sep=""))
    for(t in 1:T){
      n=n.train+1000
      X=matrix(rnorm(n*p,0.2,sqrt(0.5)),n,p)
      Z=sin((X[,1]+X[,2])*pi/10)+X[,1]^2
      Y=Z+rnorm(n,0,sqrt(0.5))
      
      
      test.idx=sample(n,1000)
      X.train=X[-test.idx,]
      Y.train=Y[-test.idx]
      Z.train=Z[-test.idx]
      X.test=X[test.idx,]
      Y.test=Y[test.idx]
      Z.test=Z[test.idx]
      write.csv(X.train,file=paste("./data/model",model1,"-",model2,"-",n.train,"/x_train_",t,".csv",sep=""))
      write.csv(Y.train,file=paste("./data/model",model1,"-",model2,"-",n.train,"/y_train_",t,".csv",sep=""))
      write.csv(X.test,file=paste("./data/model",model1,"-",model2,"-",n.train,"/x_test_",t,".csv",sep=""))
      write.csv(Y.test,file=paste("./data/model",model1,"-",model2,"-",n.train,"/y_test_",t,".csv",sep=""))
      write.csv(Z.train,file=paste("./data/model",model1,"-",model2,"-",n.train,"/z_train_",t,".csv",sep=""))
      write.csv(Z.test,file=paste("./data/model",model1,"-",model2,"-",n.train,"/z_test_",t,".csv",sep=""))
    }
    cat(model1, model2, n, "\n")
  }
  
}


#plot(Z[,2],Y^2)
#summary(lm(Y^2~I(Z[,1]^2)+I(Z[,2]^2)))
