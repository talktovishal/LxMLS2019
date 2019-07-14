import numpy as np
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = self.log_forward(input)
        #axis = 1, look at all the row values, and find the max.
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = self.backpropagation(input, output)

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]

    def log_forward(self, input):
        """Forward pass for sigmoid hidden layers and output softmax"""

        # Input
        tilde_z = input
        layer_inputs = []

        # Hidden layers
        num_hidden_layers = len(self.parameters) - 1
        for n in range(num_hidden_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Linear transformation
            weight, bias = self.parameters[n]
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation (sigmoid)
            tilde_z = 1.0 / (1 + np.exp(-z))

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Output linear transformation
        weight, bias = self.parameters[num_hidden_layers]
        z = np.dot(tilde_z, weight.T) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

        return log_tilde_z, layer_inputs

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        num_examples = input.shape[0]
        log_probability, _ = self.log_forward(input)
        return -log_probability[range(num_examples), output].mean()

    # Calculate the derivative of a sigmoid neuron output
    def my_sigmoid_derivative(self, output):
        return output * (1.0 - output)

    def ref_backpropagation(self, input, output):
        """Gradients for sigmoid hidden layers and output softmax"""

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)
        prob_y = np.exp(log_prob_y)

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1

        # For each layer in reverse store the backpropagated error, then
        # compute the gradients from the errors and the layer inputs
        errors = []

        # ----------
        # Solution to Exercise 3.2

        # Initial error is the cost derivative at the last layer (for cross
        # entropy cost)
        I = index2onehot(output, num_clases)
        error = (prob_y - I) / num_examples
        errors.append(error)

        # Backpropagate through each layer
        for n in reversed(range(num_hidden_layers)):

            # Backpropagate through linear layer
            error = np.dot(error, self.parameters[n+1][0])

            # Backpropagate through sigmoid layer
            error *= layer_inputs[n+1] * (1-layer_inputs[n+1])

            # Collect error
            errors.append(error)

        # Reverse errors
        errors = errors[::-1]

        # Compute gradients from errors
        gradients = []
        for n in range(num_hidden_layers + 1):

            # Weight gradient
            weight_gradient = np.zeros(self.parameters[n][0].shape)
            for l in range(num_examples):
                weight_gradient += np.outer(
                    errors[n][l, :],
                    layer_inputs[n][l, :]
                )

            # Bias gradient
            bias_gradient = np.sum(errors[n], axis=0, keepdims=True)

            # Store gradients
            gradients.append([weight_gradient, bias_gradient])

        # End of solution to Exercise 3.2
        # ----------

        return gradients

    def backpropagation(self, input, output):
        #print('my backprop...')
        '''
        Network definition:
        ++++++++++++++++++
        [] => node
        
               z1 ->         a1 ->       z2 ->     a2 ->                                     
        [x]       
        [x]                   [o]
        [x]                   [o]                  [OP0]
        [.]       w1          [.]         w2      
        [.]  (13989, 20)      [.]       (20,2)     [OP1]
        [.]                   [o]
        [x]

      Input(X)           Hidden Layer1        Ouptput nodes  
        13989                20                     2

        Definitions:
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

        Derivates for back-propogation:
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Let CLost = Cross Entropy loss. We want to use this loss and backpropogate and adjust
        the weights aka parameters of our model.

        1. You need to start from the last layer. Hence start with w2, b2

            Using chain rule,
            dC     dC    da2   dz2
            ---  = --- . --- . ---
            dw2    da2   dz2   dw2

            z2 = a2w2 + b3
            a2 = softmax(z2)
            dz2/dw2 = a2
            da2/dz2 = my_softmax_derivative(z2)
            dC/da2  = cost function derivative(a2) => the model code already calcualtes this for us.

            Let, a2_delta be the product of the terms below:
                       dC    da2
            a2_delta = --- . --- 
                       da2   dz2

            Note, a2_delta = error in the current code (the error-derivate to be propagated)

            dC     
            ---  = a2_delta . a2            (1)
            dw2


            For changes in biases,
            dC     dC    da2   dz2
            ---  = --- . --- . ---
            db2    da2   dz2   db2

            dz2/db2 = 1. First two terms as same from the above equation.
            Hence,
            dC
            ---  = a2_delta                 (2)
            db2    


            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            (copied from flow above, helps in chain rule)

                   z1 ->         a1 ->       z2 ->     a2 ->                                     
              w1 ->                     w2->  


          2. Now do the same for w1, b1.

            z1 = x.w1 + b1
            a1 = sigmoid(z1)

            dC     dC    da1   dz1
            ---  = --- . --- . ---
            dw1    da1   dz1   dw1

            dz1/dw1 = x
            da1/dz1 = sigmoid_derv(z1)              -- (A)

            dC     dC    da2   dz2
            ---  = --- . --- . --- => dC/da1 = a2_delta.w2 -- (B)
            da1    da2   dz2   da1

            Thus,
            dC     dC    da1   dz1
            ---  = --- . --- . ---
            dw1    da1   dz1   dw1

            and set a1_delta = dC/da1 . da1/dz1

            dC/dw1 = a1_delta * x                       -----------------   (3)

            where
            a1_delta = (a2_delta.w2) * sigmoid_derv(z1) (from A & B) --- equation (4)


            dC     dC    da1   dz1
            ---  = --- . --- . ---
            db1    da1   dz1   db1
                  = a1_delta                            -- (5)


        In this model definition of the class,
        parameters contain weights and biases.
        this is the shape:
        +		self.parameters[0][0].shape	(20, 13989)	tuple
        +		self.parameters[0][1].shape	(1, 20)	tuple
        13989 => input dimension.
        20 => hidden layer dimension.
        self.parameters[0][1].shape[0] = 1 represents the bias term. You need that for all the hidden nodes.

        +		self.parameters[1][0].shape	(2, 20)	tuple
        +		self.parameters[1][1].shape	(1, 2)	tuple

        layer inputs contain the inputs to the hidden and the output nodes
        The '30' is the batch size.
        +		layer_inputs[0].shape	(30, 13989)	tuple
        +		layer_inputs[1].shape	(30, 20)	tuple
        '''
        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)
        prob_y = np.exp(log_prob_y)

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1

        # Initial error is the cost derivative at the last layer (for cross entropy cost)
        I = index2onehot(output, num_clases)
        error = (prob_y - I) / num_examples #CE derivate

        a2 = prob_y             #output from last layer
        a1 = layer_inputs[1]
        x = layer_inputs[0]

        #why am i taking .T, since the weigths are stored as such, opposite to what i would conceptually expect
        w2 = self.parameters[1][0].T
        w1 = self.parameters[0][0].T

        a2_delta = error        #details for CE http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
        a1_delta = np.dot(a2_delta, w2.T) * self.my_sigmoid_derivative(a1) # eq(4)

        gradient_w2 = np.dot(a1.T, a2_delta)                        #eq (1)
        gradient_b2 = np.sum(a2_delta, axis=0, keepdims=True)       #eq (2)
        gradient_w1 = np.dot(x.T, a1_delta)                         #eq (3)
        gradient_b1 = np.sum(a1_delta, axis=0)                      #eq (5)
        gradients = []
        gradients.append([gradient_w1.T, np.asmatrix(gradient_b1)])
        gradients.append([gradient_w2.T, gradient_b2])
        return gradients

def main():
    import numpy as np
    import lxmls.readers.sentiment_reader as srs
    from lxmls.deep_learning.utils import AmazonData
    corpus = srs.SentimentCorpus("books")
    data = AmazonData(corpus=corpus)


    # Model
    geometry = [corpus.nr_features, 20, 2]
    activation_functions = ['sigmoid', 'softmax']

    # Optimization
    learning_rate = 0.05
    num_epochs = 10
    #batch_size = 30
    batch_size = 1

    model = NumpyMLP(
        geometry=geometry,
        activation_functions=activation_functions,
        learning_rate=learning_rate
    )

    from lxmls.deep_learning.mlp import get_mlp_parameter_handlers, get_mlp_loss_range

    # Get functions to get and set values of a particular weight of the model
    get_parameter, set_parameter = get_mlp_parameter_handlers(
        layer_index=1,
        is_bias=False,
        row=0, 
        column=0
    )

    # Get batch of data
    batch = data.batches('train', batch_size=batch_size)[0]

    # Get loss and weight value
    current_loss = model.cross_entropy_loss(batch['input'], batch['output'])
    current_weight = get_parameter(model.parameters)

    # Get range of values of the weight and loss around current parameters values
    weight_range, loss_range = get_mlp_loss_range(model, get_parameter, set_parameter, batch)

    # Get the gradient value for that weight
    gradients = model.backpropagation(batch['input'], batch['output'])
    print(gradients)
    current_gradient = get_parameter(gradients)

#local testing
if __name__ == '__main__':
    main()


'''
#not using this for now, move to comments
    # https://deepnotes.io/softmax-crossentropy
    def my_stable_softmax(self, output):
        exps = np.exp(output - np.max(output))
        return exps / np.sum(exps)

    def my_cross_entropy(self, y_hat, y):
        """
        y_hat is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
    	    Note that y is not one-hot encoded vector. 
    	    It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = self.my_stable_softmax(y_hat)
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss

    def my_cross_entropy_grad(self, X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
    	    Note that y is not one-hot encoded vector. 
    	    It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        grad = softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad

    def my_error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss


'''
