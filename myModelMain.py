import pymonntorch as pmt
import torch 
from matplotlib import pyplot as plt

import model as mdl
import current as cnt
from getDevice import get_device
from dt import TimeResolution
from plot import plot
from di import dI
from myModel import AELIF as MyAELIF


def simulate(title="LIF",
              model=mdl.LIF(),
              current=cnt.SteadyCurrent(value=6),
                DEVICE=get_device(force_cpu=True )[0],
                 dt=0.5 ,
                 iteration=1000,
                 ng_size=2,
                 print_plots = True):

      net = pmt.Network(device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)})

      ng = pmt.NeuronGroup(size= ng_size,net= net, behavior= {
                                        2: current,
                                        3: dI(),
                                        4: model,
                                        
                                        9: pmt.Recorder(variables=["u", "I","w"], tag="ng1_rec, ng1_recorder"),
                                        10: pmt.EventRecorder("spike", tag="ng1_evrec"),
                                      }
                                      )



      net.initialize()

      net.simulate_iterations(iteration)

      plot_title = f"[[{title}]]\n"
      mean_u = torch.sum(net["u", 0], axis=0) / (iteration)
      mean_I = torch.sum(net["I", 0], axis=0) / (iteration)

      plot_title += "\n".join([ f"current: {ng[2][0]}", 
                               f"model: {ng[4][0]}", 
                               f"time resolution: {dt}", 
                               f"ng size: {ng_size}", 
                               f"iteration num: {iteration}",
                               f"mean u: {mean_u}",
                               f"mean I: {mean_I}",
                                 ])


      print_plots and plot(net,plot_title)
      return net


# simulate(model=MyAELIF(w_decay_coef=0.1, di_coef=0, b=1, tau_w=10, w_booster_coef=-100, w_threshold=0), current=cnt.SteadyCurrent(value=100),iteration=1000)
simulate(model=MyAELIF(w_decay_coef=0.0, di_coef=5, b=0, tau_w=1, w_booster_coef=9, w_threshold=11), current=cnt.FlagFunction(value=100),iteration=10000)