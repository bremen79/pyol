import numpy as np

class BilinearSaddlePoint:
    def __init__(self, A, x_player, y_player):
        self.A = A
        size = np.shape(A)
        self.x_player = x_player
        self.y_player = y_player
        
    def update(self):
        x=self.x_player.get_x()
        y=self.y_player.get_x()
        grad_x=np.dot(self.A,y)
        grad_y=-np.dot(np.transpose(x),self.A)
        self.x_player.update(grad_x)
        self.y_player.update(grad_y)
        return x,y,grad_x,grad_y
                
    def get_x(self):
        return self.x_player.get_x()

    def get_y(self):
        return self.y_player.get_x()

    def get_name(self):
        return "Bilinear Game: "+self.x_player.get_name()+", "+self.y_player.get_name()

class BilinearSaddlePointAlternation(BilinearSaddlePoint):
    def update(self):
        y=self.y_player.get_x()
        grad_x=np.dot(self.A,y)
        self.x_player.update(grad_x)
        x=self.x_player.get_x()
        grad_y=-np.dot(np.transpose(x),self.A)
        self.y_player.update(grad_y)
        return x, y, grad_x, grad_y

    def get_name(self):
        return "Bilinear Game (Alternation): "+self.x_player.get_name()+", "+self.y_player.get_name()


class BilinearSaddlePointNextLoss(BilinearSaddlePoint):
    def update(self):
        # Y-player reveal its play
        y=self.y_player.get_x()
        # Show loss to X-Player and use it to update
        grad_x=np.dot(self.A,y)
        self.x_player.update(grad_x)
        x=self.x_player.get_x()
        # Y-player updates
        grad_y=-np.dot(np.transpose(x),self.A)
        self.y_player.update(grad_y)
        return x,y, grad_x, grad_y

    def get_name(self):
        return "Bilinear Game (Alternation): "+self.x_player.get_name()+", "+self.y_player.get_name()

class BilinearSaddlePointBestResponse():
    def __init__(self, A, player, best_response_index=0):
        self.A = A
        self.size = np.shape(A)
        self.best_response_index = best_response_index
        self.player = player
        self.best_response_x = np.zeros(self.size[best_response_index])
        
    def update(self):
        if self.best_response_index==0:
            # x-player is best response
            y=self.player.get_x()
            grad_x = np.dot(self.A,y)
            best_x = np.argmin(grad_x)
            x=np.zeros(self.size[0])
            x[best_x]=1
            self.best_response_x=x
            grad_y=-np.dot(np.transpose(x),self.A)
            self.player.update(grad_y)
        else:
            # y-player is best response
            x=self.player.get_x()
            grad_y = np.dot(np.transpose(x),self.A)
            best_y = np.argmax(grad_y)
            y=np.zeros(self.size[1])
            y[best_y]=1
            self.best_response_x=y
            grad_x=np.dot(self.A,y)
            self.player.update(grad_x)
        return x,y,grad_x,grad_y
            
    def get_x(self):
        if self.best_response_index==1:
            return self.player.get_x()
        else:
            return self.best_response_x
    
    def get_y(self):
        if self.best_response_index==0:
            return self.player.get_x()
        else:
            return self.best_response_x
    
    def get_name(self):
        return "Bilinear Game (Best Response): "+self.player.get_name()
