#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/sequence_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LSTMLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LSTMLayerTest() {
    vector<int> shape;
    shape.push_back(2);
    shape.push_back(3);
    shape.push_back(3);
    blob_bottom_1_ = new Blob<Dtype>(shape);
    shape.resize(1);
    blob_bottom_2_ = new Blob<Dtype>(shape);
    blob_top_ = new Blob<Dtype>();
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LSTMLayerTest() {
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_;
  }
  Blob<Dtype>* blob_bottom_1_;
  Blob<Dtype>* blob_bottom_2_;
  Blob<Dtype>* blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LSTMLayerTest, TestDtypesAndDevices);

TYPED_TEST(LSTMLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_output(4);
  shared_ptr<LSTMLayer<Dtype> > layer(
      new LSTMLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 3);
  EXPECT_EQ(this->blob_top_->count(), 24);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
  EXPECT_EQ(this->blob_top_->shape(2), 4);
}

TYPED_TEST(LSTMLayerTest, TestForwardBottomBlob2) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
    lstm_param->set_num_output(2);
    lstm_param->mutable_weight_filler()->set_type("constant");
    lstm_param->mutable_weight_filler()->set_value(1);
    lstm_param->mutable_bias_filler()->set_type("constant");
    lstm_param->mutable_bias_filler()->set_value(1);
    shared_ptr<LSTMLayer<Dtype> > layer(
        new LSTMLayer<Dtype>(layer_param));
    Dtype bottom_1_data[] = {
      1, 2, 3, 4, 5, 6, 0, 0, 0,
      2, 3, 4, 3, 4, 5, 4, 5, 6
    };
    Dtype bottom_2_data[] = {2, 3};
    this->blob_bottom_vec_[0]->set_cpu_data(bottom_1_data);
    this->blob_bottom_vec_[1]->set_cpu_data(bottom_2_data);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    EXPECT_EQ(count, 12);
    Dtype target[] = {
      0.760517, 0.760517, 0.963963, 0.963963, 0, 0,
      0.761541, 0.761541, 0.964024, 0.964024, 0.995054, 0.995054
    };
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(data[i], target[i], 1E-6);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LSTMLayerTest, TestForwardBottomBlob4) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
    lstm_param->set_num_output(2);
    lstm_param->mutable_weight_filler()->set_type("constant");
    lstm_param->mutable_weight_filler()->set_value(1);
    lstm_param->mutable_bias_filler()->set_type("constant");
    lstm_param->mutable_bias_filler()->set_value(1);
    shared_ptr<LSTMLayer<Dtype> > layer(
        new LSTMLayer<Dtype>(layer_param));
    vector<int> shape;
    shape.push_back(2);
    shape.push_back(5);
    shape.push_back(2);
    Blob<Dtype>* blob_bottom_3 = new Blob<Dtype>(shape);
    shape.resize(1);
    Blob<Dtype>* blob_bottom_4 = new Blob<Dtype>(shape);
    this->blob_bottom_vec_.push_back(blob_bottom_3);
    this->blob_bottom_vec_.push_back(blob_bottom_4);
    // I_ = 3, T_ = 3
    Dtype bottom_1_data[] = {
      1, 2, 3, 4, 5, 6, 0, 0, 0,
      2, 3, 4, 3, 4, 5, 4, 5, 6
    };
    Dtype bottom_2_data[] = {2, 3};
    // I_S_ = 2, T_S_ = 5
    Dtype bottom_3_data[] = {
      1, 2, 3, 2, 3, 4, 0, 0, 0, 0,
      4, 5, 6, 1, 2, 3, 7, 8, 9, 6
    };
    Dtype bottom_4_data[] = {3, 5};
    this->blob_bottom_vec_[0]->set_cpu_data(bottom_1_data);
    this->blob_bottom_vec_[1]->set_cpu_data(bottom_2_data);
    this->blob_bottom_vec_[2]->set_cpu_data(bottom_3_data);
    this->blob_bottom_vec_[3]->set_cpu_data(bottom_4_data);
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    EXPECT_EQ(count, 12);
    Dtype target[] = {
      0.761593, 0.761593, 0.964028, 0.964028, 0, 0,
      0.761594, 0.761594, 0.964028, 0.964028, 0.995055, 0.995055
    };
    for (int i = 0; i < count; ++i) {
      EXPECT_NEAR(data[i], target[i], 1E-6);
    }
    delete blob_bottom_3;
    delete blob_bottom_4;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LSTMLayerTest, TestGradientOutputAllBottom2) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
    lstm_param->set_num_output(2);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_min(-0.1);
    lstm_param->mutable_bias_filler()->set_max(0.1);
    // I_ = 3, T_ = 3
    Dtype bottom_1_data[] = {
      1, 2, 3, 4, 5, 6, 0, 0, 0,
      2, 3, 4, 3, 4, 5, 4, 5, 6
    };
    Dtype bottom_2_data[] = {2, 3};
    this->blob_bottom_vec_[0]->set_cpu_data(bottom_1_data);
    this->blob_bottom_vec_[1]->set_cpu_data(bottom_2_data);
    LSTMLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LSTMLayerTest, TestGradientOutputAllBottom4) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
    lstm_param->set_num_output(2);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_type("uniform");
    lstm_param->mutable_bias_filler()->set_min(-0.1);
    lstm_param->mutable_bias_filler()->set_max(0.1);
    LSTMLayer<Dtype> layer(layer_param);
    vector<int> shape;
    shape.push_back(2);
    shape.push_back(5);
    shape.push_back(2);
    Blob<Dtype>* blob_bottom_3 = new Blob<Dtype>(shape);
    shape.resize(1);
    Blob<Dtype>* blob_bottom_4 = new Blob<Dtype>(shape);
    this->blob_bottom_vec_.push_back(blob_bottom_3);
    this->blob_bottom_vec_.push_back(blob_bottom_4);
    // I_ = 3, T_ = 3
    Dtype bottom_1_data[] = {
      1, 2, 3, 4, 5, 6, 0, 0, 0,
      2, 3, 4, 3, 4, 5, 4, 5, 6
    };
    Dtype bottom_2_data[] = {2, 3};
    // I_S_ = 2, T_S_ = 5
    Dtype bottom_3_data[] = {
      1, 2, 3, 2, 3, 4, 0, 0, 0, 0,
      4, 5, 6, 1, 2, 3, 7, 8, 9, 6
    };
    Dtype bottom_4_data[] = {3, 5};
    this->blob_bottom_vec_[0]->set_cpu_data(bottom_1_data);
    this->blob_bottom_vec_[1]->set_cpu_data(bottom_2_data);
    this->blob_bottom_vec_[2]->set_cpu_data(bottom_3_data);
    this->blob_bottom_vec_[3]->set_cpu_data(bottom_4_data);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
    checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 2);
    delete blob_bottom_3;
    delete blob_bottom_4;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

// NOTE: There isn't proper funciton for this test
//TYPED_TEST(LSTMLayerTest, TestGradientOutputLastBottom2) {
//  typedef typename TypeParam::Dtype Dtype;
//  bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
//  if (Caffe::mode() == Caffe::CPU ||
//      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//    LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
//    lstm_param->set_num_output(2);
//    lstm_param->set_output_type(caffe::LSTMParameter::LAST);
//    lstm_param->mutable_weight_filler()->set_type("uniform");
//    lstm_param->mutable_bias_filler()->set_type("uniform");
//    lstm_param->mutable_bias_filler()->set_min(-0.1);
//    lstm_param->mutable_bias_filler()->set_max(0.1);
//    // I_ = 3, T_ = 3
//    Dtype bottom_1_data[] = {
//      1, 2, 3, 4, 5, 6, 0, 0, 0,
//      2, 3, 4, 3, 4, 5, 4, 5, 6
//    };
//    Dtype bottom_2_data[] = {2, 3};
//    this->blob_bottom_vec_[0]->set_cpu_data(bottom_1_data);
//    this->blob_bottom_vec_[1]->set_cpu_data(bottom_2_data);
//    LSTMLayer<Dtype> layer(layer_param);
//    GradientChecker<Dtype> checker(1e-2, 1e-3);
//    checker.CheckGradient(&layer, this->blob_bottom_vec_,
//        this->blob_top_vec_, 0);
//  } else {
//    LOG(ERROR) << "Skipping test due to old architecture.";
//  }
//}

}  // namespace caffe
