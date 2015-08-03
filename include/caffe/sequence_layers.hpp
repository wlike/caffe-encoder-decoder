#ifndef CAFFE_SEQUENCE_LAYERS_HPP_
#define CAFFE_SEQUENCE_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Processes sequential inputs using a "Long-Short Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN).
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]
 * In the implementation, the i, f, o, and g computations are performed as a
 * single inner product.
 *
 * Notably, this implementation lacks the "diagonal" gates (peephole connections),
 * as used in the LSTM architectures described by Alex Graves [3, 4] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 *
 * [4] Graves, Alex. "Supervised sequence labelling with recurrent neural networks."
 *     studies in computational intelligence, springer (2012)
 */
template <typename Dtype>
class LSTMLayer : public Layer<Dtype> {
 public:
  explicit LSTMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTM"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

 protected:
  /**
   * @param bottom input Blob vector (length 2/4)
   *
   *   -# @f$ (N \times T \times ...) @f$
   *      the time-varying input @f$ x @f$.  After the first two axes, whose
   *      dimensions must correspond to the number of independent streams
   *      @f$ N @f$ and the number of timesteps @f$ T @f$, respectively, its
   *      dimensions may be arbitrary.
   *
   *   -# @f$ (N) @f$
   *      the actual sequence length of independent streams @f$ \seq_len @f$.
   *
   *   -# @f$ (N \times T \times ...) @f$ (optional)
   *      the static (non-time-varying) input @f$ x_{static} @f$.
   *      After the first two axes, whose dimensions must correspond to the
   *      number of independent streams @f$ N @f$ and the number of timesteps
   *      @f$ T @f$, respectively, its dimensions may be arbitrary. 'Static'
   *      means at each timestep, only use the actual last timestep's info.
   *      This is mathematically equivalent to using a time-varying input of
   *      @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
   *      across the @f$ T @f$ timesteps and concatenating with the time-varying
   *      input.
   *
   *   -# @f$ (N) @f$ (optional)
   *      the actual sequence length of independent streams for static input
   *      @f$ x_{static} @f$.
   *
   * @param top output Blob vector (length 1)
   *
   *   -# @f$ (N \times T \times D) @f$
   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
   *      <code>lstm_param.num_output()</code>.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief The number of timesteps in the layer's input, and the number of
   *        timesteps over which to backpropagate through time.
   */
  int T_;
  /// @brief The number of independent streams to process simultaneously.
  int N_;
  /// @brief The dimension of input for each timestep.
  int I_;
  /// @brief The dimension of output for each timestep.
  int H_;
  /// @brief The dimension of static input for each timestep.
  int I_S_;

  /// @brief Indicate the layer's output type
  int output_type_;
  /// @brief Whether the layer has a static input copied across all timesteps.
  bool static_input_;

  /// @brief The threshold used for gradient clipping.
  Dtype grad_clip_;

  Blob<Dtype> bias_multiplier_;

  Blob<Dtype> top_;
  /// @brief gate values before nonlinearity.
  Blob<Dtype> pre_gate_;
  /// @brief gate values after nonlinearity.
  Blob<Dtype> gate_;
  /// @brief values of memory cell.
  Blob<Dtype> cell_;
  /// @brief values of tanh(memory cell).
  Blob<Dtype> tanh_cell_;

  /// @brief initial cell state value.
  Blob<Dtype> c_0_;
  /// @breif initial hidden activation value.
  Blob<Dtype> h_0_;

  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> recur_;
};

}  // namespace caffe

#endif  // CAFFE_SEQUENCE_LAYERS_HPP_
