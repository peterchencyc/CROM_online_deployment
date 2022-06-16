from abc import abstractmethod

class Experiment(object):    
    @abstractmethod
    def getPhysicalInitialCondition(self):
        pass

    @abstractmethod
    def write2file(self, time, filename, xhat):
        pass

    @abstractmethod
    def updateState(self):
        pass

    @abstractmethod
    def getResidual(self):
        pass

    @abstractmethod
    def getJacobian(self):
        pass

    @abstractmethod
    def callBack(self, step, xhat, decoder, vhat=None, xhat_prev=None):
        pass

