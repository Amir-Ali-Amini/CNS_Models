from pymonntorch import Behavior    
import math 
import torch

class MyModel(Behavior) :
    def initialize(self, ng):
        self.R = self.parameter("R" , 1)
        self.u_rest = self.parameter("u_rest", -77 ,required= True)
        self.u_reset = self.parameter("u_reset", -79,required= True)
        self.threshold = self.parameter("threshold", -55)
        self.u_rh = self.parameter("u_rh", -79,required= True)
        self.tau_m = self.parameter("tau_m", 10)
        self.delta_t = self.parameter("delta_t", 10)
        self.b = self.parameter("b", 0.1)
        self.a = self.parameter("a", 0.0)
        self.tau_w = self.parameter("tau_w", None, required=True)
        self.refractory_period = self.parameter("refractory_period" , 0) / ng.network.dt

        
        ng.refractory_time = ng.vector(-self.refractory_period-1)

        # extra parameters
        self.w_booster_coef = self.parameter("w_booster_coef", None, required=True)
        self.w_decay_coef = self.parameter("w_decay_coef", None, required=True)
        self.di_coef = self.parameter("di_coef", None, required=True)
        self.w_threshold = self.parameter("w_threshold", None)



        

        ng.u = ng.vector("uniform")*(self.threshold - self.u_reset) + self.u_reset
        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset

        ng.w = ng.vector(0.0)
        

    def forward(self, ng):
        ng.u +=  ng.network.dt * (self.Fx(ng) + self.Gx(ng))/self.tau_m

        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset

        ng.refractory_time[ng.spike] = ng.network.iteration + self.refractory_period
        self.refresh_w(ng)



    
    def Fx(self, ng):
        return self.delta_t * torch.exp((ng.u - self.u_rh)/self.delta_t) - (ng.u - self.u_rest)
    
    def Gx(self, ng):
        return self.R * (ng.I * (ng.refractory_time < ng.network.iteration).byte()  
                         +self.effect_w(ng))

    def effect_w(self, ng):
        if self.w_booster_coef :
            return (ng.w**3 / (1+ng.w**3)) * self.w_booster_coef 
        return 0
    
    def decay_w(self, ng):
        if self.w_decay_coef or self.a or self.b:
            return (-self.a * (ng.u - self.u_rest) + (-ng.w/(1+ng.w**2) )* self.w_decay_coef) + self.b * self.tau_w * ng.spike.byte()
        else: return 0
        
    def refresh_w (self, ng):
        ng.w += (self.decay_w(ng) + torch.abs(ng.di) * self.di_coef) * ng.network.dt / self.tau_w
        
        # ng.w[ng.spike] += self.b
        
        if self.w_threshold:
            ng.w[ng.w> self.w_threshold] = 0


def AELIF (threshold=+30, 
           u_rest=-65, 
           u_reset=-73.42, 
           R=1.7, 
           tau_m=10, 
           u_rh=-50, 
           delta_t=0.1, 
           a=0, 
           b=1, 
           tau_w=1,
           refractory_period = 0,
           w_booster_coef = -10,
           w_decay_coef = 0,
           di_coef = 0,
           w_threshold =60):
    return MyModel(threshold=threshold, 
                     u_reset=u_reset, 
                     u_rest=u_rest, 
                     R=R, 
                     tau_m=tau_m, 
                     u_rh=u_rh, 
                     delta_t=delta_t, 
                     a=a, 
                     b=b, 
                     tau_w=tau_w,
                     w_booster_coef = w_booster_coef,
                     w_decay_coef = w_decay_coef, 
                     di_coef = di_coef, 
                     w_threshold = w_threshold, 
                     refractory_period=refractory_period)

    