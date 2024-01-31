from gpmp_scale.bbmm.BBMM import BBMM
from gpmp_scale.gp_utils import generate_data

if __name__ == '__main__':
    # generate data
    train_x, train_y, test_x, test_y = generate_data()

    gp = BBMM(train_x, train_y)
    # plot data in chart
    gp.plot_data(train_x, train_y, test_x, test_y, image_name='BBMM_data.png')

    # Train model
    model, _ = gp.train_data(train_x, train_y)

    # Infer model
    mean, lower, upper = gp.infer(test_x, test_y)
    gp.plot_result(train_x, train_y, test_x, test_y, mean, lower, upper, file_name='BBMM_result.png', title='BBMM')
