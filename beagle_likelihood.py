from base_likelihood import BaseLikelihood

class BeagleLikelihood(BaseLikelihood):
    def __init__(self, *args, **kwargs):
       super(BeagleLikelihood, self).__init__(*args, **kwargs) 
       self.inst.make_beagle_instances(1)

