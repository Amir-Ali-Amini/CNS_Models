import pymonntorch as pmt
import torch

class SteadyCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", 6)
        self.noise_range = self.parameter("noise_range",0)
        ng.I = ng.vector(self.value)

    def forward(self,ng):
        ng.I = ng.vector(self.value) + (ng.vector("uniform")-0.5) * self.noise_range 
        # pass # no change 


class StepCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.t0 = self.parameter("t0", 200,required=True)
        self.noise_range = self.parameter("noise_range",0)
        self.value = self.parameter("value", 6)
        ng.I = ng.vector(0)

    def forward(self, ng):
        if (ng.network.iteration * ng.network.dt) > self.t0:
            ng.I = ng.vector(self.value) + (ng.vector("uniform")-0.5) * self.noise_range # increase current at t0
        else :
            ng.I = ng.vector(0) + (ng.vector("uniform")-0.5) * self.noise_range # increase current at t0


class SinCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", 6)/2
        self.noise_range = self.parameter("noise_range",0)
        self.stretch_variable= self.parameter("stretch_variable", 1)
        ng.I = ng.vector(0)

    def forward(self, ng):
        I = torch.sin(ng.vector(ng.network.iteration * ng.network.dt / self.stretch_variable)) * self.value + self.value
        ng.I = I + (ng.vector("uniform")-0.5) * self.noise_range


class UniformCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.max_current = self.parameter("value", 6) * 2
        self.step = self.parameter("step", 0.5)
        self.noise_range = self.parameter("noise_range",0)
        ng.I = ng.vector("uniform")*self.max_current

    def forward(self, ng):
        I = (ng.vector("uniform") - (ng.I/self.max_current)) * self.step
        ng.I += I + (ng.vector("uniform")-0.5) * self.noise_range




class FlagFunction(pmt.Behavior):
	def initialize(self, ng):
		self.offset = self.parameter("value")
		ng.I = ng.vector(0.)
		self.steps = self.parameter("step",1000)
		self.t = 0

	def forward(self, ng):
		if ((self.t - self.steps/2 )% self.steps == 0) :
			ng.I = ng.vector(1.)
		elif ((self.t - self.steps/2 )% self.steps == 1) :
			ng.I = ng.vector(0.)
		self.t += 1 


class IncreasingCurrent(pmt.Behavior):
	def initialize(self, ng):
		self.offset = self.parameter("value")
		self.increasing_value = self.parameter("increasing_value",0.001)
		ng.I = ng.vector(mode=self.offset)

	def forward(self, ng):
		ng.I += self.increasing_value
