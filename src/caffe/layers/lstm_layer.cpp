#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Validate bottom specification
  CHECK(bottom.size() == 2 || bottom.size() == 4);
  CHECK_EQ(bottom[0]->num_axes(), 3);
  CHECK_EQ(bottom[1]->num_axes(), 1);
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  if (bottom.size() == 4) {
    CHECK_EQ(bottom[2]->num_axes(), 3);
    CHECK_EQ(bottom[3]->num_axes(), 1);
    CHECK_EQ(bottom[2]->shape(0), bottom[3]->shape(0));
    CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0));
  }

  output_type_ = this->layer_param_.lstm_param().output_type();
  static_input_ = (bottom.size() == 4);

  T_ = bottom[0]->shape(1);
  N_ = bottom[0]->shape(0);
  I_ = bottom[0]->shape(2);
  H_ = this->layer_param_.lstm_param().num_output();
  grad_clip_ = this->layer_param_.lstm_param().grad_clip();
  if (static_input_) I_S_ = bottom[2]->shape(2);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    if (static_input_) this->blobs_.resize(4);

    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.lstm_param().bias_filler()));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.lstm_param().weight_filler()));

    // bias term
    // Initialize the bias
    vector<int> bias_shape(1, 4*H_);
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    bias_filler->Fill(this->blobs_[0].get());
 
    // hidden-to-hidden weights
    // Initialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(4*H_);
    weight_shape.push_back(H_);
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // input-to-hidden weights
    // Initialize the weight
    weight_shape.clear();
    weight_shape.push_back(4*H_);
    weight_shape.push_back(I_);
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[2].get());

    if (static_input_) {
      // input-to-hidden weights (for static input)
      // Initialize the weight
      weight_shape.clear();
      weight_shape.push_back(4*H_);
      weight_shape.push_back(I_S_);
      this->blobs_[3].reset(new Blob<Dtype>(weight_shape));
      weight_filler->Fill(this->blobs_[3].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  vector<int> cell_shape(1, H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  fdc_.Reshape(cell_shape);
  ig_.Reshape(cell_shape);
  recur_.Reshape(cell_shape);
}

template <typename Dtype>
void LSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> original_top_shape;
  original_top_shape.push_back(N_);
  original_top_shape.push_back(T_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  // Gate initialization
  vector<int> gate_shape;
  gate_shape.push_back(N_);
  gate_shape.push_back(T_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);

  vector<int> top_shape;
  top_shape.push_back(N_);
  top_shape.push_back(T_);
  top_shape.push_back(H_);
  cell_.Reshape(top_shape);
  tanh_cell_.Reshape(top_shape);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);

  // Set up the bias multiplier
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), 
    bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void LSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
  Dtype* top_data = top_.mutable_cpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
  Dtype* gate_data = gate_.mutable_cpu_data();
  Dtype* cell_data = cell_.mutable_cpu_data();
  Dtype* tanh_cell_data = tanh_cell_.mutable_cpu_data();
  Dtype* ig = ig_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_len = bottom[1]->cpu_data();
  const Dtype* bias = this->blobs_[0]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();
  const Dtype* weight_i = this->blobs_[2]->cpu_data();
  const Dtype* bottom_static_data = NULL;
  const Dtype* bottom_static_len = NULL;
  const Dtype* weight_i_s = NULL;
  if (static_input_) {
    bottom_static_data = bottom[2]->cpu_data();
    bottom_static_len = bottom[3]->cpu_data();
    weight_i_s = this->blobs_[3]->cpu_data();
  }

  // Initialize cell & hidden state
  caffe_set(c_0_.count(), (Dtype)0., c_0_.mutable_cpu_data());
  caffe_set(h_0_.count(), (Dtype)0., h_0_.mutable_cpu_data());

  // Add bias
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_*T_, 4*H_, 1, (Dtype)1.,
      bias_multiplier_.cpu_data(), bias, (Dtype)0., pre_gate_data);

  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_*T_, 4*H_, I_, (Dtype)1.,
      bottom_data, weight_i, (Dtype)1., pre_gate_data);

  for (int n = 0; n < N_; ++n) {
    int len = bottom_len[n];  // actual length of sample
    CHECK_LE(len, T_);
    int static_len = 0;
    const Dtype* x_static_t = NULL;
    if (static_input_) {
      static_len = bottom_static_len[n];
      x_static_t = bottom_static_data + bottom[2]->offset(n, static_len - 1);
    }

    // Compute recurrent forward propagation
    for (int t = 0; t < len; ++t) {
      Dtype* h_t = top_data + top_.offset(n, t);
      Dtype* c_t = cell_data + cell_.offset(n, t);
      Dtype* tanh_c_t = tanh_cell_data + tanh_cell_.offset(n, t);
      Dtype* i_t = gate_data + gate_.offset(n, t, 0);
      Dtype* f_t = gate_data + gate_.offset(n, t, 1);
      Dtype* o_t = gate_data + gate_.offset(n, t, 2);
      Dtype* g_t = gate_data + gate_.offset(n, t, 3);
      Dtype* pre_i_t = pre_gate_data + pre_gate_.offset(n, t, 0);
      Dtype* pre_g_t = pre_gate_data + pre_gate_.offset(n, t, 3);
      const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(0, 1)) : c_0_.cpu_data();
      const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(0, 1)) : h_0_.cpu_data();

      // Compute input to hidden forward propagation (for static input)
      if (static_input_) {
        caffe_cpu_gemv(CblasNoTrans, 4*H_, I_S_, (Dtype)1., weight_i_s,
            x_static_t, (Dtype)1., pre_gate_data + pre_gate_.offset(n, t));
      }

      // Hidden-to-hidden propagation
      caffe_cpu_gemv(CblasNoTrans, 4*H_, H_, (Dtype)1., weight_h, h_t_1,
          (Dtype)1., pre_gate_data + pre_gate_.offset(n, t));

      // Apply nonlinearity
      caffe_sigmoid(3*H_, pre_i_t, i_t);
      caffe_tanh(H_, pre_g_t, g_t);

      // Compute cell: c(t) = f(t) * c(t-1) + i(t) * g(t)
      caffe_mul(H_, f_t, c_t_1, c_t);
      caffe_mul(H_, i_t, g_t, ig);
      caffe_add(H_, c_t, ig, c_t);

      // Compute output: h(t) = o(t) * tanh(c(t))
      caffe_tanh(H_, c_t, tanh_c_t);
      caffe_mul(H_, o_t, tanh_c_t, h_t);
    }  // for t

    // Set padding data to zero
    if (len < T_) {
      caffe_set((T_ - len) * H_, (Dtype)0., top_data + top_.offset(n, len));
    }
  }  // for n
}

template <typename Dtype>
void LSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_len = bottom[1]->cpu_data();
  const Dtype* weight_h = this->blobs_[1]->cpu_data();
  const Dtype* weight_i = this->blobs_[2]->cpu_data();
  const Dtype* gate_data = gate_.cpu_data();
  const Dtype* cell_data = cell_.cpu_data();
  const Dtype* tanh_cell_data = tanh_cell_.cpu_data();
  const Dtype* bottom_static_data = NULL;
  const Dtype* bottom_static_len = NULL;
  const Dtype* weight_i_s = NULL;
  if (static_input_) {
    bottom_static_data = bottom[2]->cpu_data();
    bottom_static_len = bottom[3]->cpu_data();
    weight_i_s = this->blobs_[3]->cpu_data();
  }

  Dtype* top_diff = top_.mutable_cpu_diff();
  Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();
  Dtype* gate_diff = gate_.mutable_cpu_diff();
  Dtype* cell_diff = cell_.mutable_cpu_diff();

  // Set top diff to zero according to output_type
  if (caffe::LSTMParameter::LAST == (caffe::LSTMParameter::OutputType)output_type_) {
    for (int n = 0; n < N_; ++n) {
      int len = bottom_len[n];
      caffe_set((len - 1) * H_, (Dtype)0., top_diff + top_.offset(n));
    }
  }

  for (int n = 0; n < N_; ++n) {
    int len = bottom_len[n];
    int static_len = 0;
    if (static_input_) static_len = bottom_static_len[n];
    for (int t = len - 1; t >= 0; --t) {
      const Dtype* tanh_c_t = tanh_cell_data + tanh_cell_.offset(n, t);
      const Dtype* c_t = cell_data + cell_.offset(n, t);
      const Dtype* i_t = gate_data + gate_.offset(n, t, 0);
      const Dtype* f_t = gate_data + gate_.offset(n, t, 1);
      const Dtype* o_t = gate_data + gate_.offset(n, t, 2);
      const Dtype* g_t = gate_data + gate_.offset(n, t, 3);
      const Dtype* x_t = bottom_data + bottom[0]->offset(n, t);
      const Dtype* x_static_t = NULL;
      if (static_input_) {
        x_static_t = bottom_static_data + bottom[2]->offset(n, static_len-1);
      }
      Dtype* dh_t = top_diff + top_.offset(n, t);
      Dtype* dc_t = cell_diff + cell_.offset(n, t);
      Dtype* di_t = gate_diff + gate_.offset(n, t, 0);
      Dtype* df_t = gate_diff + gate_.offset(n, t, 1);
      Dtype* do_t = gate_diff + gate_.offset(n, t, 2);
      Dtype* dg_t = gate_diff + gate_.offset(n, t, 3);
      Dtype* dpre_i_t = pre_gate_diff + pre_gate_.offset(n, t, 0);
      Dtype* dpre_g_t = pre_gate_diff + pre_gate_.offset(n, t, 3);
      Dtype* fdc = fdc_.mutable_cpu_data();

      // Output gate: d_h(t) * tanh(c(t))
      caffe_mul(H_, tanh_c_t, dh_t, do_t);

      // Cell state: d_h(t) * o(t) * tanh'(c(t)) + d_c(t+1) * f(t+1)
      caffe_mul(H_, o_t, dh_t, dc_t);
      caffe_tanh_diff(H_, tanh_c_t, dc_t, dc_t);
      if (t < len - 1) {
        caffe_mul(H_, f_t + gate_.offset(0, 1), dc_t + cell_.offset(0, 1), fdc);
        caffe_add(H_, fdc, dc_t, dc_t);
      }

      // Forget gate: d_c(t) * c(t-1)
      const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(0, 1)) : c_0_.cpu_data();
      caffe_mul(H_, c_t_1, dc_t, df_t);

      // Input gate: d_c(t) * g(t)
      caffe_mul(H_, g_t, dc_t, di_t);

      // Input modulation gate: d_c(t) * i(t)
      caffe_mul(H_, i_t, dc_t, dg_t);

      // Compute derivative before nonlinearity
      caffe_sigmoid_diff(3*H_, i_t, di_t, dpre_i_t);
      caffe_tanh_diff(H_, g_t, dg_t, dpre_g_t);

      // Backpropagate errors to the previous time step
      if (t > 0) {
        Dtype* dh_t_1 = dh_t - top_.offset(0, 1);
        caffe_cpu_gemv(CblasTrans, 4*H_, H_, (Dtype)1., weight_h,
            pre_gate_diff + pre_gate_.offset(n, t), (Dtype)0.,
            recur_.mutable_cpu_data());
        caffe_add(H_, recur_.cpu_data(), dh_t_1, dh_t_1);
      }

      // Gradient w.r.t. bias
      if (this->param_propagate_down_[0]) {
        caffe_axpy(4*H_, (Dtype)1., pre_gate_diff + pre_gate_.offset(n, t),
            this->blobs_[0]->mutable_cpu_diff());
      }

      // Gradient w.r.t. hidden-to-hidden weight
      if (this->param_propagate_down_[1]) {
        if (t > 0) {
          caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 4*H_, H_, 1, (Dtype)1.,
              pre_gate_diff + pre_gate_.offset(n, t),
              top_data + top_.offset(n, t-1), (Dtype)1.,
              this->blobs_[1]->mutable_cpu_diff());
        }
      }

      // Gradient w.r.t. input-to-hidden weight
      if (this->param_propagate_down_[2]) {
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 4*H_, I_, 1, (Dtype)1.,
            pre_gate_diff + pre_gate_.offset(n, t), x_t, (Dtype)1.,
            this->blobs_[2]->mutable_cpu_diff());
      }

      // Gradient w.r.t. input-to-hidden weight (for static input)
      if (static_input_ && this->param_propagate_down_[3]) {
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 4*H_, I_S_, 1, (Dtype)1.,
            pre_gate_diff + pre_gate_.offset(n, t), x_static_t, (Dtype)1.,
            this->blobs_[3]->mutable_cpu_diff());
      }

      // Gradient w.r.t. bottom data
      if (propagate_down[0]) {
        // w.r.t. time-varing input data
        caffe_cpu_gemv(CblasTrans, 4*H_, I_, (Dtype)1., weight_i,
            pre_gate_diff + pre_gate_.offset(n, t), (Dtype)0.,
            bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n, t));
        // w.r.t. static input data
        if (static_input_) {
          caffe_cpu_gemv(CblasTrans, 4*H_, I_S_, (Dtype)1., weight_i_s,
              pre_gate_diff + pre_gate_.offset(n, t), (Dtype)1.,
              bottom[2]->mutable_cpu_diff() + bottom[2]->offset(n, static_len-1));
        }
      }
    }  // for t
  }  // for n

//  // Normalize gradients according to the number of independent streams
//  Dtype factor = (Dtype)1.0 / N_;
//  caffe_scal(this->blobs_[0]->count(), factor, this->blobs_[0]->mutable_cpu_diff());
//  caffe_scal(this->blobs_[1]->count(), factor, this->blobs_[1]->mutable_cpu_diff());
//  caffe_scal(this->blobs_[2]->count(), factor, this->blobs_[2]->mutable_cpu_diff());
//  if (static_input_) {
//    caffe_scal(this->blobs_[3]->count(), factor, this->blobs_[3]->mutable_cpu_diff());
//  }

  // Clipping gradients
  if (grad_clip_ > 0.0) {
    float norm = 0.0;
    norm += caffe_cpu_dot(this->blobs_[0]->count(),
        this->blobs_[0]->mutable_cpu_diff(), this->blobs_[0]->mutable_cpu_diff());
    norm += caffe_cpu_dot(this->blobs_[1]->count(),
        this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->mutable_cpu_diff());
    norm += caffe_cpu_dot(this->blobs_[2]->count(),
        this->blobs_[2]->mutable_cpu_diff(), this->blobs_[2]->mutable_cpu_diff());
    if (static_input_) {
      norm += caffe_cpu_dot(this->blobs_[3]->count(),
          this->blobs_[3]->mutable_cpu_diff(), this->blobs_[3]->mutable_cpu_diff());
    }
    norm = powf(norm, 0.5);
    if (norm > grad_clip_) {
      Dtype factor = grad_clip_ / norm;
      caffe_scal(this->blobs_[0]->count(), factor, this->blobs_[0]->mutable_cpu_diff());
      caffe_scal(this->blobs_[1]->count(), factor, this->blobs_[1]->mutable_cpu_diff());
      caffe_scal(this->blobs_[2]->count(), factor, this->blobs_[2]->mutable_cpu_diff());
      if (static_input_) {
        caffe_scal(this->blobs_[3]->count(), factor, this->blobs_[3]->mutable_cpu_diff());
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LSTMLayer);
#endif

INSTANTIATE_CLASS(LSTMLayer);
REGISTER_LAYER_CLASS(LSTM);

}  // namespace caffe
