"""Original TF1.12 DeepLabV3+ with fresh random initialization and accumulation."""
from __future__ import division,print_function
import json,os,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import common as c
from engine import entry,js


def erode(x):
    return tf.minimum(-tf.nn.max_pool(-x,[1,3,1,1],[1,1,1,1],'SAME'),-tf.nn.max_pool(-x,[1,1,3,1],[1,1,1,1],'SAME'))


def skeleton(x):
    s=tf.nn.relu(x-tf.nn.max_pool(erode(x),[1,3,3,1],[1,1,1,1],'SAME'))
    for _ in range(5):
        x=erode(x);d=tf.nn.relu(x-tf.nn.max_pool(erode(x),[1,3,3,1],[1,1,1,1],'SAME'));s=s+tf.nn.relu(d-s*d)
    return s


def objective(logit,y,v):
    b=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logit)
    positive=tf.boolean_mask(b,(y>.5)&(v>.5));negative=tf.boolean_mask(b,(y<.5)&(v>.5))
    k=tf.minimum(tf.size(negative),tf.maximum(1,4*tf.size(positive)+256*tf.shape(y)[0]))
    hard=tf.nn.top_k(negative,k=k).values
    bce=(tf.reduce_sum(positive)+tf.reduce_sum(hard))/tf.maximum(1.,tf.cast(tf.size(positive)+tf.size(hard),tf.float32))
    pp=tf.sigmoid(logit)*v;g=y*v;dice=1-(2*tf.reduce_sum(pp*g)+1)/(tf.reduce_sum(pp)+tf.reduce_sum(g)+1)
    sp=skeleton(pp);sg=skeleton(g);pr=(tf.reduce_sum(sp*g)+1)/(tf.reduce_sum(sp)+1);re=(tf.reduce_sum(sg*pp)+1)/(tf.reduce_sum(sg)+1)
    return bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))


class TFBackend:
    suffix='.npz';resume_suffix='.npz'
    def __init__(self,args,protocol):
        assert args.model=='deeplab';tf.set_random_seed(protocol['seed'])
        out=args.run/'deeplab';vendor=Path(protocol['tensorflow_vendor_directory'])
        original=vendor/'deeplab_model.py';assert c.sha(original)==protocol['tensorflow_vendor_SHA256']
        raw=original.read_text();before='    if is_training:\n      exclude =';assert raw.count(before)==1
        adapted=out/'deeplab_model_continuous.py'
        adapted.write_text(raw.replace(before,'    if is_training and pre_trained_model:\n      exclude ='))
        sys.path.insert(0,str(vendor));sys.path.insert(0,str(out));import deeplab_model_continuous
        self.x=tf.placeholder(tf.float32,[None,256,256,3]);self.y=tf.placeholder(tf.float32,[None,256,256,1]);self.v=tf.placeholder(tf.float32,[None,256,256,1])
        self.lr=tf.placeholder(tf.float32,[]);self.rate=protocol['initial_lr']
        net=deeplab_model_continuous.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)
        train_logits=net((self.x-.5)/.25,True);updates=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope(tf.get_variable_scope(),reuse=True):eval_logits=net((self.x-.5)/.25,False)
        self.modelvars=list(tf.global_variables())
        train_logit=(train_logits[:,:,:,1]-train_logits[:,:,:,0])[:,:,:,None]
        eval_logit=(eval_logits[:,:,:,1]-eval_logits[:,:,:,0])[:,:,:,None]
        self.probability=tf.sigmoid(eval_logit)[:,:,:,0]
        self.loss=objective(train_logit,self.y,self.v);self.eval_loss=objective(eval_logit,self.y,self.v)
        regularizer=.0001*tf.add_n([tf.nn.l2_loss(q) for q in tf.trainable_variables() if len(q.shape)>1])
        optimizer=tf.train.AdamOptimizer(self.lr)
        gradients=[(g,q) for g,q in optimizer.compute_gradients(self.loss+regularizer) if g is not None]
        accumulators=[tf.Variable(tf.zeros_like(q),trainable=False,name='gradient_accumulator_%04d'%i) for i,(g,q) in enumerate(gradients)]
        with tf.control_dependencies(updates):
            self.accumulate=tf.group(*[tf.assign_add(acc,g/protocol['gradient_accumulation']) for acc,(g,q) in zip(accumulators,gradients)])
        clipped,norm=tf.clip_by_global_norm(accumulators,5.)
        with tf.control_dependencies([tf.check_numerics(norm,'Nonfinite accumulated gradient')]):
            apply=optimizer.apply_gradients(zip(clipped,[q for g,q in gradients]))
        with tf.control_dependencies([apply]):self.apply=tf.group(*[tf.assign(acc,tf.zeros_like(acc)) for acc in accumulators])
        self.variables=list(tf.global_variables())
        self.placeholders=[tf.placeholder(q.dtype.base_dtype,q.shape) for q in self.variables]
        self.assignments=[tf.assign(q,p) for q,p in zip(self.variables,self.placeholders)]
        self.session=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads,inter_op_parallelism_threads=1,allow_soft_placement=True))
        self.session.run(tf.global_variables_initializer())
        js(out/'graph.json',dict(framework=tf.__version__,architecture='Original ResNet-v2-50 DeepLabV3+',
            vendor_SHA256=c.sha(original),adapted_SHA256=c.sha(adapted),model_variables=[q.name for q in self.modelvars],
            all_variables=[q.name for q in self.variables],random_initialization=True,public_or_project_weights_loaded=False,
            physical_batch=2,gradient_accumulation=protocol['gradient_accumulation'],batch_norm_updates_each_microbatch=True,
            regularizer=.0001,reported_loss_excludes_parameter_regularizer=True,
            initializers_only_random_graph_operations=[op.name for op in tf.get_default_graph().get_operations() if 'Random' in op.type]))
    def set_lr(self,rate):self.rate=rate
    def feed(self,x,y,v):return {self.x:x,self.y:y[:,:,:,None],self.v:v[:,:,:,None]}
    def train_batch(self,x,y,v):
        feed=self.feed(x,y,v);feed[self.lr]=self.rate
        loss,_=self.session.run([self.loss,self.accumulate],feed);return float(loss)
    def step(self):self.session.run(self.apply,{self.lr:self.rate})
    def evaluate(self,x,y,v):
        pp,loss=self.session.run([self.probability,self.eval_loss],self.feed(x,y,v));return pp,float(loss)
    def predict(self,x):return self.session.run(self.probability,{self.x:x})
    def write(self,path,variables,meta=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.npz')
        values={'v%04d'%i:q for i,q in enumerate(self.session.run(variables))}
        if meta is not None:values['metadata']=np.frombuffer(json.dumps(meta,allow_nan=False).encode('utf-8'),np.uint8)
        np.savez(str(tmp),**values);os.replace(str(tmp),str(path))
    def save_model(self,path):self.write(path,self.modelvars)
    def load_model(self,path):
        with np.load(str(path)) as weights:
            self.session.run(self.assignments[:len(self.modelvars)],
                {p:weights['v%04d'%i] for i,p in enumerate(self.placeholders[:len(self.modelvars)])})
    def save_resume(self,path,meta):self.write(path,self.variables,meta)
    def restore(self,path):
        with np.load(str(path)) as weights:
            self.session.run(self.assignments,{p:weights['v%04d'%i] for i,p in enumerate(self.placeholders)})
            return json.loads(weights['metadata'].tobytes().decode('utf-8'))
    def close(self):self.session.close()


if __name__=='__main__':entry(TFBackend)
