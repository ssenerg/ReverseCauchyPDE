from ANNPDE.PDE import ReverseChauchyPDE
from ANNPDE.PDE.shapes import (
    ElipseShape, 
    CircleShape, 
    LineShape
)
from ANNPDE.ANN import (
    LSTM, 
    laplacian,
    derivative,
    prepare_data
)
import plotly.graph_objs as go
from random import randint
from torch import nn
import numpy as np
import torch


gpu = torch.device("mps") if torch.backends.mps.is_available() else \
    torch.device("cpu")
# gpu = torch.device("cpu")
cpu = torch.device("cpu")


SEED = randint(1, 1000000)
print('Seed:', SEED)
n = 2  # Dimension of the surface
input_size = n + 1  # Input is (x1, x2, t)
hidden_layer_sizes = [10, 20, 50, 90, 150, 100, 50]  # Customize your hidden layer sizes
output_size = 1  
batch_size_divider = 512

num_epochs = 100

print(
      'Surface dimension: ', n, 
      '\nHidden layer sizes: ', hidden_layer_sizes, 
      '\nBatch size divider: ', batch_size_divider,
      '\nEpochs: ', num_epochs,
      '\n\nInput size: ', input_size, 
      '\nOutput size: ', output_size
)


F_EXPR = 'E ** x1 * sin(x2) * cos(t)'
G_EXPR = ['E ** x1 * sin(x2)', 'E ** x1 * cos(x2)']
H_EXPR = 'E ** x1 * sin(x2)'
Xs_symbol, t_symbol = ['x1', 'x2'], 't'

print()
print('f(x1, x2, t) =', F_EXPR)
print('g(x1, x2, t) =', G_EXPR)
print('h(x1, x2, t) =', H_EXPR)
T_SAMPLE = 128
E_SAMPLE = 256
D_SAMPLE = 1024
CENTER = np.array([0, 0])
RADIUS = 1

print('Time sample:', T_SAMPLE)
print('Edge sample:', E_SAMPLE)
print('Domain sample:', D_SAMPLE)
print('\nCircle Center:', CENTER)
print('Circle Radius:', RADIUS)
time = LineShape(
    seed=SEED,
    n=T_SAMPLE,
    start_point=0,
    end_point=np.pi/2,
    cross_sample_generate=1,
    even_sample=True
)
time_sample = time.get()

print('Time sample shape:', time_sample.shape)
shape = CircleShape(
    seed=SEED,
    edge_n=E_SAMPLE,
    domain_n=D_SAMPLE,
    center=CENTER,
    radius=RADIUS,
    cross_sample_generate=1,
    even_sample=True
)
edge_sample, domain_sample = shape.get()

print('Edge sample shape:', edge_sample.shape)
print('Domain sample shape:', domain_sample.shape)
model = LSTM(input_size, hidden_layer_sizes, output_size)
model.to(gpu)

criterion = nn.MSELoss()
pde = ReverseChauchyPDE(
    f_function=F_EXPR,
    g_function=G_EXPR,
    h_function=H_EXPR,
    x_symbols=Xs_symbol,
    time_symbol=t_symbol,
    criterion=criterion
)
optimizer = torch.optim.Adam(model.parameters())
domain_input = torch.from_numpy(
    prepare_data(time_sample, domain_sample)
).float().to(gpu)
edge_input = torch.from_numpy(
    prepare_data(time_sample, edge_sample)
).float().to(gpu)

domain_input = domain_input[torch.randperm(domain_input.size()[0])]
edge_input = edge_input[torch.randperm(edge_input.size()[0])]
batch_size_edge = int(edge_input.shape[0] / batch_size_divider)
batch_size_domain = int(domain_input.shape[0] / batch_size_divider)

combined_data = torch.cat((domain_input, edge_input), dim=0)

print('\nTraining Loop\n')
for epoch in range(num_epochs):

    total_tr_loss = 0.0
    total_f_loss = 0.0
    total_g_loss = 0.0
    total_h_loss = 0.0
    
    for i in range(batch_size_divider):

        batch_domain = domain_input[i*batch_size_domain:(i+1)*batch_size_domain, :].to(gpu)
        batch_edge = edge_input[i*batch_size_edge:(i+1)*batch_size_edge, :].to(gpu)

        inputs = torch.cat((batch_domain, batch_edge), dim=0)
        print('Input size: ', inputs.shape)

        outputs = model(inputs)
        print('Output size: ', outputs.shape)

        laplacian_ = laplacian(model, batch_domain)
        gradient = derivative(model, inputs)

        print(laplacian_.shape)
        print(gradient.shape)

        tr_loss = criterion(laplacian_ + gradient[:, -1], torch.zeros(laplacian_))
        print('Laplacian Loss: ', tr_loss)

        f_loss = pde.loss('f', outputs[batch_domain.size(0): ], gradient)
        print('F Loss: ', f_loss)

        g_loss = pde.loss('g', outputs[batch_domain.size(0): ], gradient) # TODO: DRICHLET O INA
        print('G Loss: ', g_loss)

        on_zero_input = torch.cat(
            (batch_domain[:, :-1], torch.zeros((batch_domain.shape[0], 1))),
            dim=1
        )
        on_zero_output = model(on_zero_input)
        h_loss = pde.loss('h', on_zero_output, on_zero_input)
        print('H Loss: ', h_loss)
        

        combined_loss = tr_loss / laplacian_.shape[0] + \
            f_loss / batch_edge.shape[0] + \
                g_loss / batch_edge.shape[0] + \
                    h_loss / batch_domain.shape[0]

        combined_loss.backward()
        optimizer.step()

        total_tr_loss += tr_loss.item()
        total_f_loss += f_loss.item()
        total_g_loss += g_loss.item()
        total_h_loss += h_loss.item()
    
    print(
        'Epoch [{}/{}], TR Loss: {:.4f}, F Loss: {:.4f}, ' \
        'G Loss: {:.4f}, H Loss: {:.4f}'.format(
            epoch+1, 
            num_epochs, 
            total_tr_loss/(i+1), 
            total_f_loss/(i+1), 
            total_g_loss/(i+1), 
            total_h_loss/(i+1)
        )
    )