#!/usr/bin/env python
# encoding=utf-8

"""
log.tracer database
choose riding data ,sort,fea11,visual ,choose valid data
generate fea [50,11]
test with wb_para


"""

c_list=['running','walking','riding','sitting','driving']

dataPath='/home/yr/ios_motion/tmpData0110/' 
test_pack=11020#01234  #012 not sure walk   |mod<15 rid  | mod>=15 walkrun|but differ between people
para_pack=0#1with liangbin riding into trainset 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
import lasagne
from leancloud import Object
from leancloud import Query
import leancloud
import time,scipy


 
inst_id='CkbmpnYPiuWoYeVNEPsjBTW7F7mnJEM9'


year_=2016
month=2;day_nov=26;
nov=[[month,day_nov,0,0,month,day_nov,23,0]]  #js 1.15-0-24  better than ios

 
##########query log.tracer
period=nov[0] #5
inst_id=inst_id
##########

def timestamp2time(timestamp):
	timestamp=float(timestamp)
	s=time.ctime(int(timestamp/1000.));#print s#'Wed Nov  4 12:43:31 2015'
	sl=s.split(' ');weekday,month,day,time24,year=sl;
	time24=time24.split(':')
	hour,minute,_=time24
	#sec=str(int(float(sll[2])*100/60.));msec=str(int(int(s[-3:])/100))
	 
	return str(hour)+'-'+str(minute)

def time2Str(moment_list):
	timeList=[]
	for timestamp_i in moment_list:
		hour_minute=timestamp2time(timestamp_i)
		print hour_minute;
		timeList.append(hour_minute)
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
	dur= [(year_, period[0], period[1], period[2], period[3], 0, 0, 0, 0),\
		(year_, period[4], period[5], period[6], period[7], 0, 0, 0, 0)]
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


 	
		
		
	 
def str2num(activity):#list
	type_list=['cycling', 'walking', 'stationary', 'running', 'automotive']
	num_list=[]
	for act in activity:
		for ind in range(type_list.__len__()):
			if act.find(type_list[ind])!=-1:
				num_list.append(ind)

			
	return np.array(num_list)
	
	
	

	
	
	

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
	stamp_range=generate_stamp(period);print 'timestamp',stamp_range


	###########
	#db log.tracer ,query location speed
	#################
	print 'query spd location...'
	log_query.equal_to('installation', inst[0]).equal_to("type",'location').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count location',log_query.count()
		 
	######get all
	spd_list=[];moment_list=[]
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
				moment=obj.get('timestamp')
				moment_list.append(moment)

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
	print 'query sensor...'
	log_query.equal_to('installation', inst[0]).equal_to("type",'sensor').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count sensor',log_query.count()
		 
	######get all
	obj_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			#print '1',obj
			
			ll=obj.get('value');#print 'll',ll None 
			if ll!=None and 'events' in ll:
				events_list=ll["events"];
				obj_list=obj_list+events_list
				print 'each event ,total obj',events_list.__len__(),obj_list.__len__()
		print 'record event',record_list.__len__(),record_list[0],obj_list.__len__()#[<object> <> <>...]
		#obj_i=record_list[0]
		data_dic={'gyro':{},'acc':{},'orientation':{},'activity':{}}
		for obj_i in obj_list:
			#for sensor in ['gyro','acc','orientation','activity']:
				#if obj_i.get('sensorName')==sensor:
			sensor_i=obj_i.get('sensorName');#print sensor_i,obj_i.get('timestamp'),obj_i.get('values')
			data_dic[sensor_i][obj_i.get('timestamp')]=obj_i.get('values')


		#print 'data dic',data_dic['acc'].__len__(),data_dic['acc'].keys()[0] 
	###########
	#db log.tracer,query motionLog
	#################
	print 'query motionLog...'
	log_query.equal_to('installation', inst[0]).equal_to("type",'motionLog').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count motionlog',log_query.count()
	######get all
	obj_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			#print 'obj',obj
			
			ll=obj.get('value');
			if ll!=None and 'events' in ll:
				#print 'get value',ll
				events_list=ll["events"];
				obj_list=obj_list+events_list; 
				print 'each event ,total obj',events_list.__len__(),obj_list.__len__()#10000events,6record
				



	 
		print 'record events',record_list.__len__(),record_list[0],obj_list.__len__()#[<object> <> <>...]
		#obj_i=record_list[0]
		#data_dic={'gyro':{},'acc':{},'orientation':{},'activity':{}}
		for obj_i in obj_list:
			#for sensor in ['gyro','acc','orientation','activity']:
				#if obj_i.get('sensorName')==sensor:
			sensor_i=obj_i.get('sensorName');#print obj_i.get('timestamp'),obj_i.get('values')
			if obj_i.get('values') !=None:
				data_dic[sensor_i][obj_i.get('timestamp')]=obj_i.get('values')

		print 'datadic',data_dic['acc'].__len__(),data_dic['acc'].keys()[0]
		 
		
	

	 	
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
	save2pickle([data_dic1,spd,moment_list],'raw-xyz')#dic array list
	   

 	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



