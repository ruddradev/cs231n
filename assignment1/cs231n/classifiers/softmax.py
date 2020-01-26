import numpy as np
from random import shuffle
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_labels=np.max(y)+1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dotp=X.dot(W)
  dotp-=np.max(dotp,axis=1,keepdims=True)
  for i in range(num_train):
     correct=dotp[i][y[i]]
     temp=np.sum(np.exp(dotp[i,:]),keepdims=False)
     loss-=np.log(np.exp(correct)/temp)
     for j in range(num_labels):
        if(j==y[i]):
            dW[:,j]+=np.transpose(X[i])*((np.exp(dotp[i][j])/temp)-1)    #Partial derivative of ith loss w.r.t the correct class weight
        else:
            dW[:,j]+=np.transpose(X[i])*((np.exp(dotp[i][j])/temp))   #Partial derivative of ith loss w.r.t the incorrect class weight
  dW/=num_train
  dW+=reg*2*W
  loss/=num_train
  loss+=reg*np.sum(np.sum(W**2))
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_naive_reference(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):

      # loss
      scores = X[i].dot(W)
      # shift values for 'scores' for numeric reasons (over-flow cautious)
      scores -= scores.max()
      scores_expsum = np.sum(np.exp(scores))
      cor_ex = np.exp(scores[y[i]])
      loss += - np.log( cor_ex / scores_expsum)

      # grad
      # for correct class
      dW[:, y[i]] += (-1) * (scores_expsum - cor_ex) / scores_expsum * X[i]
      for j in xrange(num_classes):
          # pass correct class gradient
          if j == y[i]:
              continue
          # for incorrect classes
          dW[:, j] += np.exp(scores[j]) / scores_expsum * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_labels=np.max(y)+1
  dotp=X.dot(W)
  dotp-=np.max(dotp,axis=1,keepdims=True)
  dotp=np.exp(dotp)
  correctClassExp=np.diagonal(dotp[:,y]).reshape(-1,1)
  denSum=np.sum(dotp,axis=1,keepdims=True)
  regloss=reg*np.sum(np.sum(W**2))
  dloss=(-1)*np.sum(np.log(correctClassExp/denSum),axis=0,keepdims=False)
  loss=dloss/num_train+regloss
  extraTerm=dotp/denSum    #this is the term obtained by differentiating the loss using chain rule, see naive version for example
  binmat=np.array(np.zeros_like(extraTerm))
  binmat[np.arange(num_train),y]=1  #find out which classes are correct for each image and marks them in a binary matrix
  extraTerm-=binmat  #subtracts 1 from each of those classes from the extraterm
  dW+=(np.transpose(X).dot(extraTerm))/num_train  #this dot product gives us row wise results of gradient
  dW+=reg*2*W #adding gradient of regularization
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_vectorized_reference(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # loss
  # score: N by C matrix containing class scores
  scores = X.dot(W)
  scores -= scores.max()
  scores = np.exp(scores)
  scores_sums = np.sum(scores, axis=1)
  cors = scores[range(num_train), y]
  loss = cors / scores_sums
  loss = -np.sum(np.log(loss))/num_train + reg * np.sum(W * W)

  # grad
  s = np.divide(scores, scores_sums.reshape(num_train, 1))
  s[range(num_train), y] = - (scores_sums - cors) / scores_sums
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

