#!/usr/bin/env python
# encoding=utf-8

 



dataPath='/home/yr/ios_motion_location/tmpData0121/'#for tmp data and model


test_pack=11020 
para_pack=0 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
import lasagne
from leancloud import Object
from leancloud import Query
import leancloud
import time,scipy,math
import pandas as pd


 
sensor_list=["magneticSensor","accelerometer","proximity"]

 
inst_id_jsII='ntK466fF6qCfJeYLwGYJ8od5L8n1gwXD'   # [2015.12.31
 



year=2016
month=3;day_1=4;day_2=4
nov=[[month,day_1,0,0,month,day_2,3,0]]  

 
 
#query log.tracer
period=nov[0] #5
inst_id=inst_id_jsII
##########

def timestamp2time(timestamp):
	timestamp=float(timestamp)
	s=time.ctime(int(timestamp/1000.));#print s#print s#'Wed Nov  4 12:43:31 2015'
	sl=s.split(' ');sl=[si for si in sl if len(si)>=1]#print sl;['Sat', 'Nov', '', '7', '16:54:42', '2015']
	weekday,month,day,time24,year=sl;
	time24=time24.split(':')
	hour,minute,_=time24
	#sec=str(int(float(sll[2])*100/60.));msec=str(int(int(s[-3:])/100))
	 
	#return str(hour)+'-'+str(minute)
	return float(hour)*100+float(minute)# 12.10 12.20...

def time2Str(moment_list):
	timeList=[]
	t=0
	for timestamp_i in moment_list:
		hour_minute=timestamp2time(timestamp_i)
		print 'hour minute',t,hour_minute;
		timeList.append(hour_minute)
		t+=1
	return timeList


def connect_db_log():##sensor.log.tracer   not label.sensor, 
	import leancloud,cPickle
	appid = "9ra69chz8rbbl77mlplnl4l2pxyaclm612khhytztl8b1f9o"
	   
	appkey = "1zohz2ihxp9dhqamhfpeaer8nh1ewqd9uephe9ztvkka544b"
	#appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	#appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)


def generate_stamp(period):
	#[8,28,8,33]->[(2015, 10, 20, 22, 30, 0, 0, 0, 0),(2015, 10, 20, 22, 48, 0, 0, 0, 0)]->stamp
	dur= [(year, period[0], period[1], period[2], period[3], 0, 0, 0, 0),\
		(year, period[4], period[5], period[6], period[7], 0, 0, 0, 0)]
	stamp_range0=[time2stamp(dur[0]),time2stamp(dur[1])]
	stamp_range=[t*1000 for t in stamp_range0]
	return stamp_range 

def time2stamp(t):
	#t = (2015, 9, 28, 12, 36, 38, 0, 0, 0)
	stamp = int(time.mktime( t )) ;
	return stamp
def connect_db():#label.data
	import leancloud,cPickle
	appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

 
 


#####################
def get_content(results):#result is from find() 
 	 
	obs={}; 
	r=results
	for i in range(1):
		#print type(r.get("events")) 
		if len(r.get("events"))>=1:
			 
			print r.get("motion"),r.get("events").__len__()
			ll=r.get("events") #ll=[ {},{}...]
			for dic in ll[:]:#dic={timestamp:xxxx,value:[1,2,3]...}
			
			#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
				if dic["timestamp"] not in obs.keys():
					obs[ dic["timestamp"] ]=[r.get("motion"),\
					dic["values"][0],dic["values"][1],dic["values"][2]  ]
				###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	###########################
	"""
	for k,v in obs.items():
		print k,' ',v
	"""
	print 'final',obs.__len__()

	

	#print 'i',i,count  #query-has-limit100,real-count=320
	###################3
	return obs 
	
	 


def get_all(query,skip,result):
	limit=500
	query.limit(limit)
	query.skip(skip)
	found=query.find()
	if found and len(found)>0:
		result.extend(found)
		print 'av_utils get all,now result len:',len(result),'skip',skip
		return get_all(query,skip+limit,result)
	else:
		return result
	


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	


def fea4(obs):#[50,]obs
	#4
	mean=np.mean(obs);std=np.std(obs)
	min_i=np.min(obs);max_i=np.max(obs)
	f=np.array([mean,std,min_i,max_i])#[4,]
	dim=obs.shape[0]
	#percentile 5
	percentile=[10/100.*dim,25/100.*dim,50/100.*dim,75/100.*dim,90/100.*dim];#print percentile
	perc=[int(i) for i in percentile];#print perc
	obs_sort=np.sort(obs)#[50,]
	perc_i=obs_sort[perc];#print perc_i#[5,]
	gap4=perc_i[3]-perc_i[1]
	gap5=perc_i[4]-perc_i[0]
	#sum, square-sum 12
	position=[5,10,25,75,90,95]
	pos=[int(i/100.*dim) for i in position];#print pos
	sum_i=[np.sum(obs_sort[:i]) for i in pos]#[5,]
	sqrt_sum_i=[np.sqrt(np.dot(obs_sort[:i],obs_sort[:i])) for i in pos]
	#
	fea_i=np.concatenate((f,perc_i,gap4.reshape((1,)),gap5.reshape((1,)) ),axis=0);#print fea_i.shape
	return fea_i[:]#[10,]
 



def str2num(activity):#list
	type_list=['cycling', 'walking', 'stationary', 'running', 'automotive']
	num_list=[]
	for act in activity:
		for ind in range(type_list.__len__()):
			if act.find(type_list[ind])!=-1:
				num_list.append(ind)

			
	return np.array(num_list)

 
 



def mid_smooth(mod,wind_sz):
	if mod.shape[0]<=1:return mod
	else:
		#wind_sz=10
		n=mod.shape[0];print 'n',n
		mod1=[]
		for i in range(n)[:-wind_sz]:
			patch=mod[i:i+wind_sz]#[3,]
			pi=np.sort(patch)[int(wind_sz/2)]
			mod1.append(pi)
	#
		patch=mod[-wind_sz:]
		pi=np.sort(patch)[1]
		for t in range(wind_sz):
			mod1.append(pi)
	##
	 
		print np.array(mod1).shape[0]
		return np.array(mod1)
		
	 
	
 
	

	
 
		
 
		


def timestamp2gap(timestampList):
	gapList=[0];
	for i in range(len(timestampList[1:])): #0 elem not included
		gap=(timestampList[i]-timestampList[i-1])/1000.
		gapList.append(int(gap))
		print int(gap)
		#print timestampList[i]
	return gapList
		

def location2distance(loc_list): #[dist1-0,dist2-1,dist3-2...]
	distList=[]#[0] first distance in distList is location1-location0,second distance is location2-location1
	for ind in range(len(loc_list))[1:]:
		
		x1,y1=loc_list[ind]
		x0,y0=loc_list[ind-1]
		dist=( (x1-x0)*(x1-x0)+(y1-y0)*(y1-y0) )
		dist=math.sqrt(dist)
		distList.append(dist);
 	distList.append(0)
	return distList 

def getSpd(distList,gapList):
	spdList=[];meanMinute=sum(gapList)/float(len(gapList))
	minObs=min(len(distList),len(gapList))
	for i in range(minObs):
		spd=distList[i]/float(np.abs(gapList[i])+1.)*meanMinute
		spdList.append(spd)
	return spdList
		

def location2distance_origine(loc_list):#[dist0-0,dist1-0,dist2-0...]
	distList=[0]#[0]first distance in list is 0,second distance is location1-location0
	for ind in range(len(loc_list))[1:]:
		
		x1,y1=loc_list[ind]
		x0,y0=loc_list[0]
		dist=( (x1-x0)*(x1-x0)+(y1-y0)*(y1-y0) )
		dist=math.sqrt(dist)
		distList.append(dist);
	#distList.append(0)
 
	return distList 


def fill100(xyz): #[101,3] [98,3]
	n,d=xyz.shape
	if n>=100:return xyz[:100,:]
	if n<100:
		while n<100:
			xyz=np.concatenate((xyz,xyz),axis=0)
			n,d=xyz.shape
		return xyz[:100,:]
	
			
def getMinuteGap(timestampList):
	minuteGapList=[0]
	for i in range(len(timestampList))[1:]:
		t1=int(timestampList[i]/1000)
		t0=int(timestampList[i-1]/1000)
		g=t1-t0  
		minuteGapList.append(int(g/60.) )
	 
	return minuteGapList
	 
		
def generateX(acc_mod3):
	###acc walk(ride) drive sit run
	fea=acc_mod3;print ' generate fea acc',fea.shape
	#kernel_sz=50.;stride=kernel_sz; #when training,cluster denoise windowSize=50
	kernel_sz=100.;stride=kernel_sz;
	obs_list=[]; 
	num=int( (fea.shape[0]-kernel_sz)/stride ) +1
    	for i in range(num)[:]: #[0,...100] total 101 
        	obs100=fea[stride*i:i*stride+kernel_sz]#[100,]
		if obs100.shape[0]==kernel_sz: #[100,]
			obs3=split_obs(obs100)#[100,]->[ [50,],[50,],[50,]  ]  overlap 3obs
			obs3=[ fea4(obs_i) for obs_i in obs3 ]#[ [11,],[11,],[11,] ] 3obs
			obs_list=obs_list+obs3# [11,]
			
	x_arr_acc=np.array(obs_list);print 'x tobe tested',x_arr_acc.shape
	return x_arr_acc
def split_obs(obs_fea4):  
	if len(obs_fea4.shape)==1: #[100,]->[[50,],[50,],[50,]  ]
		return [obs_fea4[0:50],obs_fea4[25:75],obs_fea4[50:]  ]
	elif len(obs_fea4.shape)==2:#[100,2]->[[50,2],[50,2],[50,2]  ]
		return [obs_fea4[0:50,:],obs_fea4[25:75,:],obs_fea4[50:,:]  ]	


def predictX(x_arr_acc):
	########
	#load model
	#########
	class_dic={'walking':0,'driving':1,'sitting':2,'running':3}
	bdt_acc=load_pickle(dataPath+'bdt0112')#acc sit drive walk run
	
	#########
	#acc  floor 1  walk drive sit run
	########
	y_pred3=bdt_acc.predict(x_arr_acc);print 'pred y',y_pred3.shape
	y_pred=np.array(voteEvery3(y_pred3)  );print y_pred.shape#[51,]->[17,] list  not work put here
	return y_pred

def voteEvery3(y_pred):#[51,]->[17,]
	def majorityVote(arr):#[3,]->1x1
		uniq=np.unique(arr)
		if uniq.shape[0]==3 or uniq.shape[0]==1:return uniq[0]
		elif uniq.shape[0]==2:#for example [3,1,1]
			vote1=len(np.where(arr==uniq[0])[0])
			vote2=len(np.where(arr==uniq[1])[0])
			return [uniq[0] if vote1>vote2 else uniq[1] ][0]

	##############
	ker=3.;stri=ker
	 
	pred_list=[]; 
	num=int( (y_pred.shape[0]-ker)/stri ) +1
    	for i in range(num)[:]: #
        	vote=y_pred[stri*i:i*stri+ker]#[3,]
		if vote.shape[0]==ker: #[3,]
			vote=majorityVote(vote)#[3,]->1x1
			pred_list.append(vote)
	return pred_list
	



 

		 
	

############3
if __name__=="__main__":
	

	       
	######################################
	#load from db log SPEED
	#########################################
	
	 
	####init
	connect_db_log()
	log = leancloud.Object.extend('Log')
	log_query = leancloud.Query(log)
	#print 'all',log_query.count()##error
	inst_query = leancloud.Query(leancloud.Installation)
	print 'install',inst_query.count()#2335
	inst = inst_query.equal_to('objectId', inst_id).find();#print '1',inst[0]
	#
	
	#################each period
	all_record_list=[];i=0
	
	 
	
	####period
	stamp_range=generate_stamp(period);print stamp_range


	###########
	#db log.tracer ,query location speed
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'location').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count location',log_query.count()
		 
	######get all
	spd_list=[];loc_list=[];timestampLocList=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			#print '1',obj
			
			ll=obj.get('value');#print 'll',ll None 
			if ll!=None and 'speed' in ll:
				spd=ll["speed"]
				spd_list.append(spd)
			###
			loc=obj.get('location')
			timestamp=obj.get('timestamp') 
			loc_list.append([loc.latitude,loc.longitude])
			timestampLocList.append(timestamp)
			##
			 
	

	#
	spd=np.array(spd_list);print 'spd',spd.shape#[n,]
	
	###################
	#
	#####################
	#plt.figure()
	#plt.title('spd-'+class_type)
    	#plt.plot(np.array(spd),'bo');#plt.xlim(0,x_axis);
	#
	

	 
	###########
	#db log.tracer,query sensor
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'sensor').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count sensor',log_query.count()
		 
	######get all
	obj_list=[];timestampSensorList=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:#each record
			#print '1',obj
			
			ll=obj.get('value');#print 'll',ll None 
			if ll!=None and 'events' in ll:
				#
				events_list=ll["events"];
				obj_list=obj_list+events_list
				if len(events_list)>200:timestampSensorList.append( obj.get('timestamp') )
				print 'each event ,total obj',len(timestampSensorList),events_list.__len__(),obj_list.__len__()
				
		print 'record event',record_list.__len__(),record_list[0],obj_list.__len__()#[<object> <> <>...]
		#obj_i=record_list[0]
		data_dic={'gyro':{},'acc':{},'orientation':{},'activity':{}}
		dd=1/float(len(obj_list));di=0
		for obj_i in obj_list:
			di+=1. #sampe timestamp different value
			#for sensor in ['gyro','acc','orientation','activity']:
				#if obj_i.get('sensorName')==sensor:
			sensor_i=obj_i.get('sensorName');#print sensor_i,obj_i.get('timestamp'),obj_i.get('values')
			data_dic[sensor_i][obj_i.get('timestamp')+dd*di]=obj_i.get('values')


		print 'acc',data_dic['acc'].__len__() 



	###########
	#db log.tracer,query motionLog
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'motionLog').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count motionlog',log_query.count()
	######get all
	obj_list=[];timestampMotionList=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			#print 'obj',obj
			
			ll=obj.get('value');
			if ll!=None and 'events' in ll:
				#
				#print 'get value',ll
				events_list=ll["events"];
				obj_list=obj_list+events_list; 
				#
				if len(events_list)>200:#skip those with 'ios-activity' but not sensor data
					timestampMotionList.append( obj.get('timestamp') )
				print 'each event ,total obj',len(timestampMotionList),events_list.__len__(),obj_list.__len__()#10000events,6record
				



	 
		print 'record events',record_list.__len__(),record_list[0],obj_list.__len__()#[<object> <> <>...]
		#obj_i=record_list[0]
		#data_dic={'gyro':{},'acc':{},'orientation':{},'activity':{}}
		dd=1/float(len(obj_list));di=0
		for obj_i in obj_list: #[ {},...]   {}={time:,value:,sensorname:...}
			di+=1.
			#for sensor in ['gyro','acc','orientation','activity']:
				#if obj_i.get('sensorName')==sensor:
			sensor_i=obj_i.get('sensorName');#print obj_i.get('timestamp'),obj_i.get('values')
			if obj_i.get('values') !=None:
				data_dic[sensor_i][obj_i.get('timestamp')+dd*di]=obj_i.get('values')

		print 'acc',data_dic['acc'].__len__(),data_dic['acc'].keys()[0]
		 
		
	

	 	
	### 
	#sort by timestamp
	data_dic1={}
	for sensor,v in data_dic.items():
		if sensor!='activity':
			print 'sensor',sensor,v.__len__() 
			if len(v)>1:
				ll=sorted(v.items(),key=lambda f:f[0],reverse=False)
				# # DATA FORMATE  {timestamp:[x y z],...}  ->  [ (timestamp,[x,y,z]),...]
				xyz=np.array([obs[1] for obs in ll]);print xyz.shape#[n,3]
				data_dic1[sensor]=xyz
			else: data_dic1[sensor]=np.zeros((10,3))
		else:
			ll=sorted(v.items(),key=lambda f:f[0],reverse=False)#{timestamp:{v},...} {v}='walking':0.25
			activity=[obs[1].keys()[0] for obs in ll];print 'activity',set(activity),activity.__len__(), 
			data_dic1[sensor]=str2num(activity)#[n,]
	################
	 

	 
	###############################################
	print 'spd  loc  timestampLoc  ',spd.shape,len(loc_list),len(timestampLocList) 
	print 'acc',data_dic1['acc'].shape,len(timestampSensorList),len(timestampMotionList) 
	timestampList=timestampSensorList+timestampMotionList
	timestampList.sort()
	print len(timestampList)
	save2pickle([data_dic1,loc_list,timestampList,timestampLocList],'raw-xyz')
	save2pickle(data_dic1['acc'],'xyz-acc-0201')
	   
	   

 	    
	######################3
	#clean data:match [ acc_time ,location_time] locationTime less
	########################
	data_dic1,loc_list,timestampList,timestampLocList=load_pickle(dataPath+'raw-xyz');
	print 'acc time',data_dic1['acc'].shape,len(timestampList)
	print 'loc time',len(timestampLocList),len(loc_list)

	#
	timestampLoc=[t/1. for t in timestampLocList]
	timestampSensor=[t/1. for t in timestampList]
	

	 		
				

	#timeListLoc=time2Str(timestampLoc)#hourMinute list
	timeListLoc=time2Str(timestampList);print len(timeListLoc)
	 


	 
	###############
	# mod  acc mag gyro
	##################
	
	xyz_acc=data_dic1['acc']
	acc_mod=np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) );print '1',acc_mod.shape#[n,]
	acc_mod=mid_smooth(acc_mod,3)
	#orient_mod=np.sqrt( (xyz[:,3:6]*xyz[:,3:6]).sum(axis=1) );
	#gyro_mod=np.sqrt( (xyz[:,6:]*xyz[:,6:]).sum(axis=1) );#[n,]
	plt.figure()
	#plt.title('acc mag gyro mod')
	#mod
	#plt.subplot(3,1,3);plt.title('gyro mod'+class_type);plt.plot(gyro_mod,'bo');plt.ylim(0,5);#plt.xlim(0,1000)
	#plt.subplot(3,1,2);plt.title('orient mag mod');plt.plot(orient_mod,'bo');#plt.xlim(0,1000)
	plt.subplot(3,1,1);plt.title('acc mod');plt.plot(acc_mod,'r-');plt.ylim(0,3);plt.grid()#plt.xlim(0,1000)
	#xyz acc gyro 
	plt.figure()
	sensor_list=['acc','orientation','gyro']
	color=['r-','g-','b-']
	
	plt.subplot(3,1,1);
	plt.title('acc xyz');#plt.xlim(0,200);plt.ylim(-5,5)
	for axis in range(3):
		plt.plot( xyz_acc[:,axis] ,color[axis])

	plt.subplot(3,1,2);
	plt.title('mag xyz');#plt.xlim(0,200);plt.ylim(-5,5)
	for axis in range(3):
		plt.plot(np.abs( data_dic1['orientation'][:,axis] ),color[axis])

	plt.subplot(3,1,3);
	plt.title('gyro xyz');plt.ylim(-0.5,0.5)#plt.xlim(0,200);
	for axis in range(3):
		plt.plot(data_dic1['gyro'][:,axis] ,color[axis])


	 


 
	 
			
	    

	 
	#motion 
	class_dic={'walking':0,'driving':1,'sitting':2,'running':3}
	xarr=generateX(acc_mod) #[nx100,]->[n,11]fea
	pred_y=predictX(xarr) ;print 'motion',pred_y.shape
	strList=[] 
	for y in pred_y:
		for k,v in class_dic.items():
			if v==y:strList.append(k)


	preds_out = pd.DataFrame({'timestamp':timestampList,"predict": strList})
	preds_out.to_csv(dataPath+"timestamp_predict.csv")	
		
	

	 
	 

	 
	plt.show()
	
	 
	 
	
	 
	 

	

	 
	 
	 



 

	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



