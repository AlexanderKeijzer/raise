use inline_python::python;
use crate::tensor::Tensor;

pub fn imshow(tensor: &Tensor, im_shape: Option<[usize; 2]>) {
    assert!(tensor.shape[3] == 1);
    let v = tensor.to_vec();
    let mut s = im_shape.unwrap_or_else(|| [tensor.shape[0], tensor.shape[1]]).to_vec();
    if tensor.shape[3] > 1 {
        s.push(tensor.shape[3])
    }
    python! {
        import numpy as np
        import matplotlib.pyplot as plt

        img = np.array('v).reshape('s)
        plt.imshow(img)
        plt.show()
    }
}

pub fn plot1d(vec: Vec<f32>) {
    python! {
        import matplotlib.pyplot as plt

        plt.plot('vec)
        plt.show()
    }
}

pub fn plot2d(x: Vec<f32>, y: Vec<f32>) {
    python! {
        import matplotlib.pyplot as plt

        plt.plot('x, 'y)
        plt.show()
    }
}

pub fn plot2dlog(x: Vec<f32>, y: Vec<f32>) {
    python! {
        import matplotlib.pyplot as plt

        plt.plot('x, 'y)
        plt.xscale("log")
        plt.show()
    }
}