#include <algorithm>
#include <cctype>
#include <locale>
#include <iostream>
#include <fstream>
#include <chrono>
#include <system_error>
#include <sstream>
#include "sparsehash/dense_hash_map"

#include <stdio.h>
#include <sched.h>
#include <errno.h>

/*
inline uint64_t getTime() {
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
}
*/

/**
 * cpu_pin - pin down the local thread to a core
 * @cpu: the target core
 */
void cpu_pin(unsigned int cpu)
{
  int ret;
  cpu_set_t mask;

  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);

  ret = sched_setaffinity(0, sizeof(mask), &mask);
  if (ret) throw std::system_error(errno, std::system_category());
}

// Credit: https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
      return !std::isspace(ch);
  }).base(), s.end());
}

/**
 * Check that the power governor for a core is "performance",
 * logging a warning if it is not.
 */
void check_cpu_governor(unsigned int cpu)
{
  std::ostringstream fpss;
  fpss << "/sys/devices/system/cpu/cpu" << cpu << "/cpufreq/scaling_governor";
  std::ifstream f(fpss.str());
  if (!f.is_open()) {
    std::cerr << "WARNING: Could not find CPU " << cpu << " governor information in filesystem (are you running on Linux?)\n";
    std::cerr << "The file '" << fpss.str() << "' did not exist.\n";
  }
  std::ostringstream r;
  r << f.rdbuf();
  std::string gov(r.str());
  rtrim(gov);
  if (gov != "performance") {
    std::cerr << "WARNING: CPU " << cpu << " governor is " << gov << ", which could lead to variance in performance.\n";
    std::cerr << "Run 'echo performance > " << fpss.str() << "' as root to turn off power scaling.\n";
  }
}

#ifndef NO_PYTHON
#include <Python.h>

#define REQUIRE JIT_ASSERT

#else

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#endif

#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/attributes.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include "torch/csrc/assertions.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/jit/passes/shape_analysis.h"

#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/module.h"

#include <vector>
#include <iostream>

uint64_t MakeTime(struct timespec const& ts) {
  return ts.tv_sec*1000000000LL + ts.tv_nsec;
}

inline uint64_t getTime() {
  struct timespec ts;
  JIT_ASSERT(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0);
  return MakeTime(ts);
}

#ifndef NO_PYTHON
#include "torch/csrc/utils/auto_gil.h"
#else
struct AutoNoGIL {};
#endif

namespace torch { namespace jit {

using Var = SymbolicVariable;

using namespace torch::autograd;

template<typename T>
static std::ostream & operator<<(std::ostream & out, const std::vector<T> & list) {
  size_t i = 0;
  out << "{";
  for(auto && e : list) {
    if(i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}
static auto ct = CodeTemplate(R"(
int foo($args) {

    $bar
        $bar
    $a+$b
}
int commatest(int a${,stuff})
int notest(int a${,empty,})
)");
static auto ct_expect = R"(
int foo(hi, 8) {

    what
    on many
    lines...
    7
        what
        on many
        lines...
        7
    3+4
}
int commatest(int a, things..., others)
int notest(int a)
)";

static void codeTemplateTest() {
  {
    TemplateEnv e;
    e.s("hi","foo");
    e.v("what",{"is","this"});
    TemplateEnv c(e);
    c.s("hi","foo2");
    REQUIRE(e.s("hi") == "foo");
    REQUIRE(c.s("hi") == "foo2");
    REQUIRE(e.v("what")[0] == "is");
  }

  {
    TemplateEnv e;
    e.v("args",{"hi","8"});
    e.v("bar",{"what\non many\nlines...","7"});
    e.s("a","3");
    e.s("b","4");
    e.v("stuff",{"things...","others"});
    e.v("empty",{});
    auto s = ct.format(e);
    //std::cout << "'" << s << "'\n";
    //std::cout << "'" << ct_expect << "'\n";
    REQUIRE(s == ct_expect);
  }
}

Value * appendNewNode(NodeKind kind, Graph& graph, ArrayRef<Value*> inputs) {
  return graph.appendNode(graph.create(kind,inputs))->output();
}


static void fusionTests() {
  FusionCompiler comp;

  auto testSimple = [&] {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    auto a = at::rand(at::CUDA(at::kFloat), {3,4});
    auto b = at::rand(at::CUDA(at::kFloat), {4,3}).transpose(0,1);
    auto o = at::zeros(at::CUDA(at::kFloat), {3,4});
    comp.debugLaunchGraph(graph, 0, {a,b}, {o});
    auto o2 = a*b;
    float max_diff = (o2 - o).abs().max().toCDouble();
    //std::cout << "max diff: " << max_diff << "\n";
    REQUIRE(max_diff == 0);
  };
  testSimple();

  auto testOne = [&](int ti, int tj, int toi, int toj) {

    Graph graph;

    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    Var i2 = Var::asNewInput(graph);
    Var i3 = Var::asNewInput(graph);
    Var i4 = Var::asNewInput(graph);

    auto p22 =  i4.sigmoid();
    auto p20 = i3.sigmoid();
    auto p18 = i2.tanh();
    auto p16 = i1.sigmoid();
    auto p14 = p20 * i0;
    auto p11 = p22 * p18;
    auto o1 = p14 + p11;
    auto p5 = o1.tanh();
    auto o0 = p16 * p5;
    o0.addAsOutput();
    o1.addAsOutput();

    graph.lint();

    std::vector<at::Tensor> inputs;
    std::vector<at::Tensor> outputs;
    // We want to generate input/output tensors with dimension 128x128x32, but
    // with different internal strides.  To do this, we generate a tensor
    // with the "wrong" dimensions, and then use transpose to get an appropriately
    // sized view.
    for(size_t i = 0; i < graph.inputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[ti],dims[tj]);
      inputs.push_back(at::rand(at::CUDA(at::kFloat), dims).transpose(ti, tj));
    }
    for(size_t i = 0; i < graph.outputs().size(); i++) {
      std::vector<int64_t> dims = {128, 128, 32};
      std::swap(dims[toi],dims[toj]);
      outputs.push_back(at::zeros(at::CUDA(at::kFloat), dims).transpose(toi,toj));
    }

    auto t22 = inputs[4].sigmoid();
    auto t20 = inputs[3].sigmoid();
    auto t18 = inputs[2].tanh();
    auto t16 = inputs[1].sigmoid();
    auto t14 = t20*inputs[0];
    auto t11 = t22*t18;
    auto out1 = t14+t11;
    auto t5 = out1.tanh();
    auto out0 = t16*t5;


    //auto out0 = inputs[0]*inputs[1];
    comp.debugLaunchGraph(graph, 0, inputs, outputs);
    REQUIRE(out0.is_same_size(outputs.front()));
    float max_diff = (outputs.front() - out0).abs().max().toCDouble();
    REQUIRE(max_diff < 1e-6);

  };
  testOne(0,0,0,0);
  testOne(0,1,0,0);
  testOne(1,2,0,0);
  testOne(0,2,0,0);

  testOne(0,0,0,1);
  testOne(0,1,1,2);
  testOne(1,2,0,2);



  auto testConcat = [&](int dim) {
    Graph graph;
    Var i0 = Var::asNewInput(graph);
    Var i1 = Var::asNewInput(graph);
    auto o0 = i0 * i1;
    o0.addAsOutput();
    Var::cat({i0, o0}, dim).addAsOutput();

    auto a = at::rand(at::CUDA(at::kFloat), {3,4,5});
    auto b = at::rand(at::CUDA(at::kFloat), {4,3,5}).transpose(0,1);
    auto o = at::zeros(at::CUDA(at::kFloat), {3,4,5});

    auto o_r = a*b;
    auto o2_r = at::cat({a, o_r}, dim);
    auto o2 = at::zeros(at::CUDA(at::kFloat), o2_r.sizes());
    comp.debugLaunchGraph(graph, 0, {a,b}, {o, o2});

    float max_diff = (o_r - o).abs().max().toCDouble();
    REQUIRE(max_diff == 0);
    float max_diff2 = (o2_r - o2).abs().max().toCDouble();
    REQUIRE(max_diff2 == 0);
  };
  testConcat(0);
  testConcat(1);
  testConcat(2);
}

struct Attr : public Attributes<Attr> {
};
void attributesTest() {
  auto one = attr::alpha;
  auto two = attr::device;
  auto three = attr::end;
  auto four = attr::perm;
  Attr attr;
  attr.f_(one,3.4)->i_(two,5)->s_(three,"what");
  REQUIRE(attr.f(one) == 3.4);
  REQUIRE(attr.s(three) == "what");
  REQUIRE(attr.i(two) == 5);
  attr.s_(one,"no");
  REQUIRE(attr.s(one) == "no");
  REQUIRE(attr.hasAttribute(three));
  REQUIRE(!attr.hasAttribute(four));
  attr.ss_(two, {"hi", "now"});
  REQUIRE(attr.ss(two).at(1) == "now");

  Attr attr2;
  attr2.copyAttributes(attr);
  REQUIRE(attr2.s(one) == "no");
  attr2.f_(one,5);
  REQUIRE(attr.s(one) == "no");
  REQUIRE(attr2.f(one) == 5);
}

void internedStringsTests () {

  REQUIRE(prim::Param == Symbol::prim("Param"));
  REQUIRE(prim::Return == Symbol::prim("Return"));
  REQUIRE(prim::Return.toUnqualString() == std::string("Return"));
  REQUIRE(prim::Return.toQualString() == std::string("prim::Return"));
  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  size_t symstart = newsym;
  REQUIRE(newsym.toQualString() == std::string("aten::__NEW_SYMBOL"));
  // TODO: This test is a bit too close to the implementation details.
  REQUIRE(Symbol::aten("What") == symstart+1);
  REQUIRE(Symbol::aten("What2") == symstart+2);
  REQUIRE(Symbol::aten("What") == symstart+1);
  REQUIRE(Symbol::aten("What2") == symstart+2);
  REQUIRE(Symbol(SymbolNamespace::aten, symstart+2).toUnqualString() == std::string("What2"));
}

void fromQualStringTests() {
  REQUIRE(Symbol::fromQualString("prim::Param") == Symbol::prim("Param"));
  REQUIRE(Symbol::fromQualString("aten::mm") == Symbol::aten("mm"));
  REQUIRE(Symbol::fromQualString("onnx::LSTM") == Symbol::onnx("LSTM"));
  REQUIRE(Symbol::fromQualString("attr::value") == Symbol::attr("value"));
  REQUIRE(Symbol::fromQualString("scope::") == Symbol::scope(""));
  auto bad_inputs = {"scope", "foo::bar", "prim:Param", "::", ":", ""};
  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      REQUIRE(0);
    } catch (std::runtime_error c) {
    }
  }
}

at::Tensor t_use(at::Tensor x) {
  return x;
}
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

// given the difference of output vs expected tensor, check whether the
// difference is within a relative tolerance range. This is a standard way of
// matching tensor values upto certain precision
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().toCFloat(), maxValue);
  }
  return diff.abs().max().toCFloat() < 2e-5 * maxValue;
}
bool almostEqual(const at::Tensor & a, const at::Tensor & b) {
  return checkRtol(a - b,{a, b});
}

bool exactlyEqual(const at::Tensor & a, const at::Tensor & b) {
  return (a - b).abs().max().toCFloat() == 0.f;
}

std::pair<at::Tensor, at::Tensor>
lstm(at::Tensor input,
      at::Tensor hx,
      at::Tensor cx,
      at::Tensor w_ih,
      at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));
  auto chunked_gates = gates.chunk(4, 1);
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

std::tuple<Var, Var> build_lstm_body(
  Graph & g,
  Var input,
  Var hx,
  Var cx,
  Var w_ih,
  Var w_hh) {
    auto gates = input.mm(w_ih);
    gates = gates + hx.mm(w_hh);
    auto outputs = gates.chunk(4, 1);
    auto ingate = outputs[0];
    auto forgetgate = outputs[1];
    auto cellgate = outputs[2];
    auto outgate = outputs[3];
    ingate = ingate.sigmoid();
    outgate = outgate.sigmoid();
    cellgate = cellgate.tanh();
    forgetgate = forgetgate.sigmoid();

    auto cy = forgetgate*cx;
    cy =  cy + ingate*cellgate;
    auto hy = outgate*cy.tanh();

    return std::make_tuple(hy,cy);
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto & g = *r;
  Value * input = g.addInput();
  Value * hx = g.addInput();
  Value * cx = g.addInput();
  Value * w_ih = g.addInput();
  Value * w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

std::shared_ptr<Graph> build_lstm_stages() {
  auto r = std::make_shared<Graph>();
  auto & g = *r;
  Var input = g.addInput();
  Var hx = g.addInput();
  Var cx = g.addInput();
  Var w_ih = g.addInput();
  Var w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  // use some stuff from the previous stage as well
  // as a new input
  g.advanceStage();
  hx = hy;
  cy.addAsOutput();
  cx = g.addInput();

  std::tie(hy,cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
}

void runOneStage(InterpreterState & interp, const std::vector<at::Tensor> & inputs, std::vector<at::Tensor> & outputs) {
  outputs = inputs;
  interp.runOneStage(outputs);
}

void interpTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;

    auto input = at::randn(at::CUDA(at::kFloat), {seq_len, batch_size, input_size});
    auto hx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
    auto cx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
    auto w_ih  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, input_size}));
    auto w_hh  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, hidden_size}));

    auto lstm_g = build_lstm();
    Code lstm_function(lstm_g, /*values_are_variables=*/false);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    runOneStage(lstm_interp, {input[0], hx, cx, w_ih, w_hh}, outputs);
    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    REQUIRE(exactlyEqual(outputs[0],hx));
    REQUIRE(exactlyEqual(outputs[1],cx));
}

void interpStageTest() {
    constexpr int batch_size = 4;
    constexpr int input_size = 256;
    constexpr int seq_len = 32;

    int hidden_size = 2*input_size;
    auto input = at::randn(at::CUDA(at::kFloat), {seq_len, batch_size, input_size});
    auto hx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
    auto cx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
    auto cx1 = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
    auto w_ih  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, input_size}));
    auto w_hh  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, hidden_size}));


    auto lstm_g = build_lstm_stages();
    Code lstm_function(lstm_g, /*values_are_variables=*/false);
    std::vector<at::Tensor> outputs;
    InterpreterState lstm_interp(lstm_function);
    runOneStage(lstm_interp, {input[0], hx, cx, w_ih, w_hh}, outputs);
    auto cy0 = outputs[0];
    runOneStage(lstm_interp, {cx1}, outputs);
    at::Tensor ihx = outputs[0];
    at::Tensor icx = outputs[1];


    std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
    std::tie(hx, cx) = lstm(input[0], hx, cx1, w_ih, w_hh);

    //std::cout << almostEqual(outputs[0],hx) << "\n";
    REQUIRE(exactlyEqual(outputs[0],hx));
    REQUIRE(exactlyEqual(outputs[1],cx));
}

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(const char *name, var_meta_list input_meta, test_fn_type test_fn)
    : name(name)
    , input_meta(input_meta)
    , test_fn(test_fn) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto & m : input_meta) {
      out.emplace_back(autograd::make_variable(at::CPU(at::kFloat).tensor(m).normal_(), /*requires_grad=*/true));
    }
    return out;
  }

  const char *name;
  var_meta_list input_meta;
  test_fn_type test_fn;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
                      return v.type().tensor(v.sizes()).normal_();
                    });
}

std::shared_ptr<Graph> trace(const ADTestSpec& test, const variable_list& vars_in) {
  std::shared_ptr<tracer::TracingState> state;
  variable_list trace_vars_in;
  std::tie(state, trace_vars_in) = tracer::enter(vars_in, 1);
  auto trace_vars_out = test(trace_vars_in);
  tracer::exit(trace_vars_out);
  return state->graph;
}

variable_list grad(const variable_list& outputs, const variable_list& inputs, const variable_list& grad_outputs) {
  static const auto get_edge = [](const Variable& v) { return v.gradient_edge(); };
  auto & engine = torch::autograd::Engine::getDefaultEngine();
  return engine.execute(fmap(outputs, get_edge), grad_outputs, true, false, fmap(inputs, get_edge));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].is_same_size(b[i]));
    REQUIRE(a[i].allclose(b[i]));
  }
}

std::pair<tensor_list, tensor_list> runGradient(Gradient& grad_spec,
                                                tensor_list& tensors_in,
                                                tensor_list& tensor_grads_in) {
  tensor_list tensors_out, tensor_grads_out;
  Code f_code{grad_spec.f, /*values_are_variables=*/false},
      df_code{grad_spec.df, /*values_are_variables=*/false};
  InterpreterState f_interpreter { f_code }, df_interpreter { df_code };

  runOneStage(f_interpreter, tensors_in, tensors_out);

  tensor_list df_inputs;
  df_inputs.insert(df_inputs.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  for(auto offset : grad_spec.df_input_captured_inputs)
    df_inputs.push_back(tensors_in[offset]);
  for(auto offset : grad_spec.df_input_captured_outputs)
    df_inputs.push_back(tensors_out[offset]);
  runOneStage(df_interpreter, df_inputs, tensor_grads_out);

  // Outputs of f needs to be sliced
  tensors_out.erase(tensors_out.begin() + grad_spec.f_real_outputs, tensors_out.end());
  return std::make_pair(tensors_out, tensor_grads_out);
}

void testADFormulas() {
  static const auto unwrap = [](const Variable& v) { return v.data(); };

  using VL = variable_list;
  static const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  static const var_meta_list unary_pointwise  = {{2, 3, 4, 5}};
  static const std::vector<ADTestSpec> ad_tests = {
    {"add",     binary_pointwise, [](const VL& v) -> VL { return {v[0] + v[1]}; }},
    {"sub",     binary_pointwise, [](const VL& v) -> VL { return {v[0] - v[1]}; }},
    {"mul",     binary_pointwise, [](const VL& v) -> VL { return {v[0] * v[1]}; }},
    {"sigmoid", unary_pointwise,  [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
    {"tanh",    unary_pointwise,  [](const VL& v) -> VL { return {v[0].tanh()}; }},
    {"t",       unary_pointwise,  [](const VL& v) -> VL { return {v[0].t()}; }},
    {"mm",      {{10, 12}, {12, 15}}, [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
    {"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].chunk(4, 1)); }},
    {"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].chunk(3, 2)); }},
    {"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].split(4, 1)); }},
    {"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return fmap<Variable>(v[0].split(3, 2)); }},
  };

  // We have to release the GIL inside this method, because if we happen to
  // initialize the autograd engine here, the newly spawned worker threads will
  // try to initialize their PyThreadState*, and they need the GIL for this.
  AutoNoGIL _no_gil;
  for (const auto & test : ad_tests) {
    // Get reference values form autograd
    auto vars_in        = test.make_vars();
    auto vars_out       = test(vars_in);
    auto var_grads_in   = get_grad_outputs(vars_out);
    auto var_grads_out  = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = trace(test, vars_in);
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    auto grad_spec = differentiate(graph, std::vector<bool>(vars_in.size(), true));

    // Get outputs from the interpreter
    auto tensors_in                = fmap(vars_in, unwrap);
    auto tensor_grads_in           = fmap(var_grads_in, unwrap);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) = runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out      = fmap(vars_out, unwrap);
    auto expected_tensor_grads_out = fmap(var_grads_out, unwrap);
    assertAllClose(tensors_out,      expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

std::string toString(std::shared_ptr<Graph>& graph) {
  std::ostringstream s;
  s << *graph;
  return s.str();
}

void testDifferentiate(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = std::shared_ptr<TensorType>(new TensorType(s, -1, {2, 3, 4}, {12, 4, 1}));

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b * a + b;
  graph->registerOutput(c.value());

  auto grad_spec = differentiate(graph, {true, true});
  std::vector<std::size_t> expected_captured_inputs = {0, 1};
  std::vector<std::size_t> expected_captured_outputs = {1};
  std::vector<std::size_t> expected_input_vjps = {0, 1};
  std::vector<std::size_t> expected_output_vjps = {0, 1};
  REQUIRE(grad_spec.f_real_outputs == 1);
  REQUIRE(grad_spec.df_input_captured_inputs == expected_captured_inputs);
  REQUIRE(grad_spec.df_input_captured_outputs == expected_captured_outputs);
  REQUIRE(grad_spec.df_input_vjps == expected_input_vjps);
  REQUIRE(grad_spec.df_output_vjps == expected_output_vjps);
  out << "testDifferentiate\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testDifferentiateWithRequiresGrad(std::ostream & out) {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = std::shared_ptr<TensorType>(new TensorType(s, -1, {2, 3, 4}, {12, 4, 1}));

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto d = b * b + b;
  auto e = (d + a) * a + b;
  graph->registerOutput(d.value());
  graph->registerOutput(e.value());

  auto grad_spec = differentiate(graph, {true, false});
  std::vector<std::size_t> expected_input_vjps = {1, 2};  // for e and %4 = (d + a)
  std::vector<std::size_t> expected_output_vjps = {0};    // only a requires grad
  REQUIRE(grad_spec.f_real_outputs == 2);              // we need one temporary %4 = (d + a)
  REQUIRE(grad_spec.df_input_captured_inputs == std::vector<std::size_t>({0}));
  REQUIRE(grad_spec.df_input_captured_outputs == std::vector<std::size_t>({2}));
  REQUIRE(grad_spec.df_input_vjps == expected_input_vjps);
  REQUIRE(grad_spec.df_output_vjps == expected_output_vjps);
  out << "testDifferentiateWithRequiresGrad\n";
  out << *grad_spec.f;
  out << *grad_spec.df;
  out << "\n";
}

void testCreateAutodiffSubgraphs(std::ostream & out) {
  auto graph = build_lstm();
  CreateAutodiffSubgraphs(*graph, /*threshold=*/2);
  out << "testCreateAutodiffSubgraphs\n";
  out << *graph << "\n";
}

autograd::Variable var(at::Type & t, at::IntList sizes, bool requires_grad) {
  return autograd::make_variable(at::rand(t, sizes), requires_grad);
}
autograd::Variable undef() {
  return autograd::Variable();
}

int device(const autograd::Variable & v) {
  return v.type().is_cuda() ? v.get_device() : -1;
}

bool isEqual(at::IntList lhs, at::IntList rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool isEqual(const TensorInfo & ti, const autograd::Variable & v) {
  if(!ti.defined())
    return ti.defined() == v.defined();
  return
    ti.device() == device(v) &&
    ti.requires_grad() == v.requires_grad() &&
    ti.type() == v.type().scalarType() &&
    isEqual(ti.sizes(), v.sizes()) &&
    isEqual(ti.strides(), v.strides());
}

// work around the fact that variable_tensor_list doesn't duplicate all
// of std::vector's constructors.
// most constructors are never used in the implementation, just in our tests.
variable_tensor_list createVarList(std::vector<at::Tensor> && list) {
  return variable_tensor_list(std::move(list));
}

void argumentSpecTest() {
  auto & CF = at::CPU(at::kFloat);
  auto & CD = at::CPU(at::kDouble);
  auto & GF = at::CUDA(at::kFloat);
  auto & GD = at::CUDA(at::kDouble);

  auto list =  createVarList({ var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()});

  // make sure we have some non-standard strides
  list[1].transpose_(0, 1);

  // same list but different backing values
  auto list2 = createVarList({ var(CF, {1}, true), var(CD, {1, 2}, false) , var(GF, {}, true), var(GD, {4,5,6}, false), undef()});
  list2[1].transpose_(0, 1);


  ArgumentSpec a(true, list);
  ArgumentSpec b(true, list);
  REQUIRE(a.hashCode() == b.hashCode());

  REQUIRE(a == b);
  ArgumentSpec d(true, list2);
  REQUIRE(d == a);
  REQUIRE(d.hashCode() == a.hashCode());

  for(size_t i = 0; i < list.size(); ++i) {
    REQUIRE(isEqual(a.tensorInfo(i), list[i]));
  }
  ArgumentSpec no_grad(/*with_grad=*/false, list);
  REQUIRE(no_grad != a);

  std::unordered_set<ArgumentSpec> spec;
  spec.insert(std::move(a));
  REQUIRE(spec.count(b) > 0);
  REQUIRE(spec.count(no_grad) == 0);
  spec.insert(std::move(no_grad));
  REQUIRE(spec.count(ArgumentSpec(true,list)) == 1);

  list2[1].transpose_(0,1);
  ArgumentSpec c(true, list2); // same as list, except for one stride
  REQUIRE(!(c == a));
  REQUIRE(spec.count(c) == 0);

}

void shapeAnalysisTest() {

  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2*input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn(at::CUDA(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, input_size}));
  auto w_hh  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, hidden_size}));

  auto g = build_lstm();
  ArgumentSpec spec(false, createVarList({v(input), v(hx), v(cx), v(w_ih), v(w_hh) }));
  PropagateInputShapes(*g, spec);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  auto o0 = g->outputs()[0]->type()->expect<TensorType>();
  auto o1 = g->outputs()[1]->type()->expect<TensorType>();
  REQUIRE(o0->sizes() == std::vector<int64_t>(r0.sizes().begin(), r0.sizes().end()));
  REQUIRE(o1->sizes() == std::vector<int64_t>(r1.sizes().begin(), r1.sizes().end()));

}

void testGraphExecutor() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2*input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn(at::CUDA(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CUDA(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, input_size}));
  auto w_hh  = t_def(at::randn(at::CUDA(at::kFloat), {4 * hidden_size, hidden_size}));

  std::vector<at::Tensor> inputs = {v(input), v(hx), v(cx), v(w_ih), v(w_hh) };
  auto g = build_lstm();
  GraphExecutor executor(g);
  auto outputs = executor.run(variable_tensor_list(std::move(inputs)));
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  REQUIRE(almostEqual(Variable(outputs[0]).data(), r0));
  REQUIRE(almostEqual(Variable(outputs[1]).data(), r1));
}

void testBlocks(std::ostream & out) {
  Graph g;
  auto a = Var::asNewInput(g, "a");
  auto b = Var::asNewInput(g, "b");
  auto c = a + b;
  auto r = g.appendNode(g.create(prim::If, {Var::asNewInput(g, "c").value()}));
  auto then_block = r->addBlock();
  auto else_block = r->addBlock();
  {
    WithInsertPoint guard(then_block);
    auto t = c + c;
    then_block->registerOutput(t.value());
  }
  {
    WithInsertPoint guard(else_block);
    auto  d = b + c;
    auto e = d + c;
    else_block->registerOutput(e.value());
  }
  g.registerOutput((Var(r->output()) + c).value());
  g.lint();
  out << "testBlocks\n" << g << "\n";
  r->eraseBlock(0);
  out << g << "\n";
  g.lint();
  // test recursive copy of blocks works
  auto g2 = g.copy();
  out << *g2 << "\n";
}


const static auto cf_examples = R"JIT(
  def if_test(a, b):
      c = 0
      if a < b:
        c = b
      else:
        c = a
      return c
  def if_one(a, b):
    c = b
    if a < b:
      c = a
    return c
  def while_test(a, i):
    while i < 3:
      a *= a
      i += 1
    return a
)JIT";
void testControlFlow() {
  script::Module cu;
  script::defineMethodsInModule(cu, cf_examples, torch::jit::script::Resolver(), nullptr);
  auto run = [&](const std::string & name, std::vector<at::Tensor> stack) {
    auto graph = cu.get_method(name).graph();
    Code code(graph, /*values_are_variables=*/false);
    InterpreterState interp(code);
    interp.runOneStage(stack);
    return stack;
  };

  auto F = [](float f) { return at::Scalar(f).toTensor(); };
  auto V = [](at::Tensor t) { return at::Scalar(t).toFloat(); };
  auto run_binary = [&](const std::string & name, float a, float b) {
    return V(run(name, {F(a), F(b)})[0]);
  };
  REQUIRE(2 == run_binary("if_test", 1, 2));
  REQUIRE(3 == run_binary("if_test", 3, 2));
  REQUIRE(2 == run_binary("if_one", 2, 3));
  REQUIRE(2 == run_binary("if_one", 3, 2));
  REQUIRE(256 == run_binary("while_test",2,0));
}

#ifdef NO_PYTHON

TEST_CASE( "jit test CPU", "[cpu]" ) {

  std::stringstream out;
  SECTION( "control flow" )
    testControlFlow();
  SECTION( "blocks" )
    testBlocks(out);
  SECTION( "create autodiff subgraphs" )
    testCreateAutodiffSubgraphs(out);
  SECTION( "differentiate" )
    testDifferentiate(out);
  SECTION( "differentiate with requires grad" )
    testDifferentiateWithRequiresGrad(out);
  SECTION( "AD formulas" )
    testADFormulas();
  SECTION( "code template" )
    codeTemplateTest();
  SECTION( "attributes" )
    attributesTest();
  SECTION( "interned strings" )
    internedStringsTests();
}

TEST_CASE( "jit test CUDA", "[cuda]" ) {

  SECTION( "graph executor" )
    testGraphExecutor();
  SECTION( "fusion" )
    fusionTests();
  SECTION( "interp" )
    interpTest();
  SECTION( "interp stage" )
    interpStageTest();
  SECTION( "argument spec" )
    argumentSpecTest();
  SECTION( "shape analysis" )
    shapeAnalysisTest();
}

#endif


__attribute__((noinline))
int flushCache(int foo) {
  constexpr int _30meg = 30*1024*1024;
  auto a = new char[_30meg];
  for(int i = 0; i < _30meg; i++) {
    a[i] = i*foo;
  }
  int r = a[rand() % _30meg];
  delete [] a;
  return r;
}

// mm, chunk, sigmoid, tanh, +, *

constexpr int batch_size = 1;
constexpr int input_size = 128;
constexpr int hidden_size = 128;


namespace direct {

struct Ten;

Ten plus_f(const Ten& a, const Ten& b);
Ten mul_f(const Ten& a, const Ten& b);
Ten sigmoid_f(const Ten& a);
Ten tanh_f(const Ten& a);
Ten mm_f(const Ten& a, const Ten& b);
std::vector<Ten> chunk4_f(const Ten& a);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);

struct TenImpl : at::Retainable {
  int64_t sizes[2];
  float * data;
  ~TenImpl() {
    delete [] data;
  }
};


// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }

  bool defined() const {
    return pImpl;
  }
  Ten operator+(const Ten& b) const {
    return plus_f(*this, b);
  }
  Ten operator*(const Ten& b) const {
    return mul_f(*this, b);
  }
  Ten mm(const Ten& b) const {
    return mm_f(*this, b);
  }
  Ten sigmoid() const {
    return sigmoid_f(*this);
  }
  Ten tanh() const {
    return tanh_f(*this);
  }
  std::vector<Ten> chunk4() {
    return chunk4_f(*this);
  }
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(const Ten& a, const Ten& b) {
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(const Ten& a) {
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void doit() {

  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }

  std::cout << "direct " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}

namespace vtable {

struct Ten;

Ten plus_f(const Ten& a, const Ten& b);
Ten mul_f(const Ten& a, const Ten& b);
Ten sigmoid_f(const Ten& a);
Ten tanh_f(const Ten& a);
Ten mm_f(const Ten& a, const Ten& b);
std::vector<Ten> chunk4_f(const Ten& a);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);
#define CONCAT_(x,y) x##y
#define CONCAT(x,y) CONCAT_(x,y)
#define FNAME() CONCAT(foo,__LINE__)

struct TenImpl : at::Retainable {
  int64_t sizes[2];
  float * data;
  // add a ton of fake vtable entries to simulate perf when all operators exist
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual Ten plus(const Ten& a, const Ten& b);
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual Ten mul(const Ten& a, const Ten& b);
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual Ten sigmoid(const Ten& a);
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual Ten tanh(const Ten& a);
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual Ten mm(const Ten& a, const Ten& b);
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual void FNAME()() {
    std::cout << __LINE__ << "\n";
  }
  virtual std::vector<Ten> chunk4(const Ten& a);

  ~TenImpl() {
    delete [] data;
  }
};


// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }

  bool defined() const {
    return pImpl;
  }
  Ten operator+(const Ten& b) const {
    return pImpl->plus(*this, b);
  }
  Ten operator*(const Ten& b) const {
    return pImpl->mul(*this, b);
  }
  Ten mm(const Ten& b) const {
    return pImpl->mm(*this, b);
  }
  Ten sigmoid() const {
    return pImpl->sigmoid(*this);
  }
  Ten tanh() const {
    return pImpl->tanh(*this);
  }
  std::vector<Ten> chunk4() {
    return pImpl->chunk4(*this);
  }
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

 Ten TenImpl::plus(const Ten& a, const Ten& b) {
  return plus_f(a, b);
}
 Ten TenImpl::mul(const Ten& a, const Ten& b) {
  return mul_f(a, b);
}
 Ten TenImpl::sigmoid(const Ten& a) {
  return sigmoid_f(a);
}
 Ten TenImpl::tanh(const Ten& a) {
  return tanh_f(a);
}
Ten TenImpl::mm(const Ten& a, const Ten& b) {
    return mm_f(a,b);
}
std::vector<Ten> TenImpl::chunk4(const Ten& a) {
  return chunk4_f(a);
}

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(const Ten& a, const Ten& b) {
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(const Ten& a) {
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void doit() {

  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }
  std::cout << "vtable " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}

namespace custom_vtable {

struct Ten;

Ten plus_f(const Ten& a, const Ten& b);
Ten mul_f(const Ten& a, const Ten& b);
Ten sigmoid_f(const Ten& a);
Ten tanh_f(const Ten& a);
Ten mm_f(const Ten& a, const Ten& b);
std::vector<Ten> chunk4_f(const Ten& a);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);

struct TenImpl : at::Retainable {
  void ** vtable;
  int64_t sizes[2];
  float * data;
  ~TenImpl() {
    delete [] data;
  }
};

enum {
  ADD = 66,
  MUL = 133,
  SIGMOID = 200,
  TANH = 266,
  MM = 333,
  CHUNK = 400
};


// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }

  bool defined() const {
    return pImpl;
  }
  Ten operator+(const Ten& b) const {
    auto fn = (decltype(&plus_f))(this->pImpl->vtable[ADD]);
    return fn(*this, b);
  }
  Ten operator*(const Ten& b) const {
    auto fn = (decltype(&mul_f))(this->pImpl->vtable[MUL]);
    return fn(*this, b);
  }
  Ten mm(const Ten& b) const {
    auto fn = (decltype(&mm_f))(this->pImpl->vtable[MM]);
    return fn(*this, b);
  }
  Ten sigmoid() const {
    auto fn = (decltype(&sigmoid_f))(this->pImpl->vtable[SIGMOID]);
    return fn(*this);
  }
  Ten tanh() const {
    auto fn = (decltype(&tanh_f))(this->pImpl->vtable[TANH]);
    return fn(*this);
  }
  std::vector<Ten> chunk4() {
    auto fn = (decltype(&chunk4_f))(this->pImpl->vtable[CHUNK]);
    return fn(*this);
  }
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

static void** the_vtable = nullptr;

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->vtable = the_vtable;
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(const Ten& a, const Ten& b) {
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(const Ten& a) {
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void doit() {
  the_vtable = new void*[401];
  the_vtable[ADD] = (void*)&plus_f;
  the_vtable[MUL] = (void*)&mul_f;
  the_vtable[SIGMOID] = (void*)&sigmoid_f;
  the_vtable[TANH] = (void*)&tanh_f;
  the_vtable[MM] = (void*)&mm_f;
  the_vtable[CHUNK] = (void*)&chunk4_f;

  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }

  std::cout << "custom_vtable " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}

// what if we use the absolutely simplest, slowest way of dispatch through a hash table?
namespace slowhash {

struct Ten;

Ten plus_f(const Ten& a, const Ten& b);
Ten mul_f(const Ten& a, const Ten& b);
Ten sigmoid_f(const Ten& a);
Ten tanh_f(const Ten& a);
Ten mm_f(const Ten& a, const Ten& b);
std::vector<Ten> chunk4_f(const Ten& a);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);


const int32_t cpu_float_tensor_key = int(at::kFloat);

struct TenImpl : at::Retainable {
  int64_t sizes[2];
  float * data;
  int32_t hash_key;
  ~TenImpl() {
    delete [] data;
  }
};

enum {
  ADD = 66,
  MUL = 133,
  SIGMOID = 200,
  TANH = 266,
  MM = 333,
  CHUNK = 400
};

std::unordered_map<std::string, void*> dispatch_table;

std::string get_key(at::ArrayRef<int32_t> key) {
  return std::string((char*)key.data(), (char*)(key.data() + key.size()));
}

template<typename T> T lookup(at::ArrayRef<int32_t> key) {
  return (T)(dispatch_table[get_key(key)]);
}

// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }
  int32_t key() const {
    return pImpl->hash_key;
  }
  bool defined() const {
    return pImpl;
  }
  Ten operator+(const Ten& b) const {
    return lookup<decltype(&plus_f)>({ADD, key(), b.key()})(*this, b);
  }
  Ten operator*(const Ten& b) const {
    return lookup<decltype(&mul_f)>({MUL, key(), b.key()})(*this, b);
  }
  Ten mm(const Ten& b) const {
    return lookup<decltype(&mm_f)>({MM, key(), b.key()})(*this, b);
  }
  Ten sigmoid() const {
      return lookup<decltype(&sigmoid_f)>({SIGMOID, key()})(*this);
  }
  Ten tanh() const {
    return lookup<decltype(&tanh_f)>({TANH, key()})(*this);
  }
  std::vector<Ten> chunk4() {
      return lookup<decltype(&chunk4_f)>({CHUNK, key()})(*this);
  }
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->hash_key = cpu_float_tensor_key;
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(const Ten& a, const Ten& b) {
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(const Ten& a) {
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void noop() {
  std::cout << "NOP\n";
}

void doit() {

  // make up some operators
  for(int scalar_type = 0; scalar_type < 8; scalar_type++) {
    for(int backend = 0; backend < 4; backend++) {
      int32_t hash_key = scalar_type*4 + backend;
      for(int op = 0; op < 401; op++) {
        if(op % 2 == 0) {
          dispatch_table[get_key({op, hash_key})] = (void*)noop;
        } else {
          dispatch_table[get_key({op, hash_key, hash_key})] = (void*)noop;
        }
      }
    }
  }
  // register the real ones...
  dispatch_table[get_key({ADD,cpu_float_tensor_key,cpu_float_tensor_key})] = (void*) plus_f;
  dispatch_table[get_key({MUL,cpu_float_tensor_key,cpu_float_tensor_key})] = (void*)mul_f;
  dispatch_table[get_key({MM,cpu_float_tensor_key,cpu_float_tensor_key})] = (void*)mm_f;
  dispatch_table[get_key({SIGMOID,cpu_float_tensor_key})] = (void*)sigmoid_f;
  dispatch_table[get_key({TANH,cpu_float_tensor_key})] = (void*)tanh_f;
  dispatch_table[get_key({CHUNK,cpu_float_tensor_key})] = (void*)chunk4_f;

  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }

  std::cout << "slowhash " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}

// still really bad hash table, but keep the hash key a constant size
namespace consthash {

struct Ten;

Ten plus_f(ArrayRef<Ten>);
Ten mul_f(ArrayRef<Ten>);
Ten sigmoid_f(ArrayRef<Ten>);
Ten tanh_f(ArrayRef<Ten>);
Ten mm_f(ArrayRef<Ten>);
std::vector<Ten> chunk4_f(ArrayRef<Ten>);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);


const int32_t cpu_float_tensor_key = int(at::kFloat);

struct HashKey {
  HashKey()
  : inside{0, 0, 0, 0}
  , rest(nullptr) {
  }
  HashKey(int32_t a, int32_t b)
  : inside{a, b, 0, 0}
  , rest(nullptr) {
  }
  HashKey(int32_t a, int32_t b, int32_t c)
  : inside{a, b, c, 0}
  , rest(nullptr) {
  }
  int32_t inside[4];
  std::string* rest;
  bool operator==(const HashKey& rhs) const {
    for(int i = 0; i < 4; i++) {
      if(inside[i] != rhs.inside[i]) {
        return false;
      }
    }
    if(rest && rhs.rest)
      return *rest == *rhs.rest;
    return rest == rhs.rest;
  }
};

struct HashKeyHasher {
  std::size_t operator () (const HashKey &key) const {
    size_t seed = 0;
    int64_t* data = (int64_t*) key.inside;
    for(int i = 0; i < 2; i++) {
      seed = torch::hash_combine(seed, data[i]);
    }
    return seed;
  }
};


//static_assert(sizeof(HashKey) == 256/8, "wrong size");

struct TenImpl : at::Retainable {
  int64_t sizes[2];
  float * data;
  int32_t hash_key;
  ~TenImpl() {
    delete [] data;
  }
};

enum {
  ADD = 66,
  MUL = 133,
  SIGMOID = 200,
  TANH = 266,
  MM = 333,
  CHUNK = 400
};

google::dense_hash_map<HashKey, void*, HashKeyHasher> dispatch_table;

// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }
  int32_t key() const {
    return pImpl->hash_key;
  }
  bool defined() const {
    return pImpl;
  }
  #define LOOKUP(fn, ...) ((decltype(&fn))(dispatch_table[HashKey(__VA_ARGS__)]))
  Ten operator+(const Ten& b) const {
    return LOOKUP(plus_f, ADD, key(), b.key())({*this, b});
  }
  Ten operator*(const Ten& b) const {
    return LOOKUP(mul_f, MUL, key(), b.key())({*this, b});
  }
  Ten mm(const Ten& b) const {
    return LOOKUP(mm_f, MM, key(), b.key())({*this, b});
  }
  Ten sigmoid() const {
    return LOOKUP(sigmoid_f, SIGMOID, key())({*this});
  }
  Ten tanh() const {
    return LOOKUP(tanh_f, TANH, key())({*this});
  }
  std::vector<Ten> chunk4() {
    return LOOKUP(chunk4_f, CHUNK, key())({*this});
  }
  #undef LOOKUP
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->hash_key = cpu_float_tensor_key;
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(at::ArrayRef<Ten> v) {
  auto a = v[0];
  auto b = v[1];
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(at::ArrayRef<Ten> v) {
  auto a = v[0];
  auto b = v[1];
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(at::ArrayRef<Ten> v) {
  auto a = v[0];
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(at::ArrayRef<Ten> v) {
  auto a = v[0];
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(at::ArrayRef<Ten> v) {
  auto a = v[0];
  auto b = v[1];
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(ArrayRef<Ten> v) {
  auto a = v[0];
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void noop() {
  std::cout << "NOP\n";
}

void doit() {
  dispatch_table.set_empty_key(HashKey(10000,0));
  // make up some operators
  for(int scalar_type = 0; scalar_type < 8; scalar_type++) {
    for(int backend = 0; backend < 4; backend++) {
      int32_t hash_key = scalar_type*4 + backend;
      for(int op = 0; op < 401; op++) {
        if(op % 2 == 0) {
          dispatch_table[HashKey(op, hash_key)] = (void*)noop;
        } else {
          dispatch_table[HashKey(op, hash_key, hash_key)] = (void*)noop;
        }
      }
    }
  }
  // register the real ones...
  dispatch_table[HashKey(ADD,cpu_float_tensor_key,cpu_float_tensor_key)] = (void*) plus_f;
  dispatch_table[HashKey(MUL,cpu_float_tensor_key,cpu_float_tensor_key)] = (void*)mul_f;
  dispatch_table[HashKey(MM,cpu_float_tensor_key,cpu_float_tensor_key)] = (void*)mm_f;
  dispatch_table[HashKey(SIGMOID,cpu_float_tensor_key)] = (void*)sigmoid_f;
  dispatch_table[HashKey(TANH,cpu_float_tensor_key)] = (void*)tanh_f;
  dispatch_table[HashKey(CHUNK,cpu_float_tensor_key)] = (void*)chunk4_f;


  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }

  std::cout << "consthash " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}

namespace customhash {

struct Ten;

Ten plus_f(const Ten& a, const Ten& b);
Ten mul_f(const Ten& a, const Ten& b);
Ten sigmoid_f(const Ten& a);
Ten tanh_f(const Ten& a);
Ten mm_f(const Ten& a, const Ten& b);
std::vector<Ten> chunk4_f(const Ten& a);
Ten load(const at::Tensor& a);
at::Tensor save(const Ten& a);


const int32_t cpu_float_tensor_key = int(at::kFloat);

struct HashKey {
  HashKey()
  : inside{0, 0, 0, 0}
  , rest(nullptr) {
  }
  HashKey(int32_t a, int32_t b)
  : inside{a, b, 0, 0}
  , rest(nullptr) {
  }
  HashKey(int32_t a, int32_t b, int32_t c)
  : inside{a, b, c, 0}
  , rest(nullptr) {
  }
  int32_t inside[4];
  std::string* rest;
  bool operator==(const HashKey& rhs) const {
    for(int i = 0; i < 4; i++) {
      if(inside[i] != rhs.inside[i]) {
        return false;
      }
    }
    if(rest && rhs.rest)
      return *rest == *rhs.rest;
    return rest == rhs.rest;
  }
  bool empty() {
    return inside[0] == 0;
  }
  size_t hash() const {
    size_t seed = 0;
    int64_t* data = (int64_t*) inside;
    for(int i = 0; i < 2; i++) {
      seed = torch::hash_combine(seed, data[i]);
    }
    return seed;
  }
};


//static_assert(sizeof(HashKey) == 256/8, "wrong size");

struct TenImpl : at::Retainable {
  int64_t sizes[2];
  float * data;
  int32_t hash_key;
  ~TenImpl() {
    delete [] data;
  }
};

enum {
  ADD = 66,
  MUL = 133,
  SIGMOID = 200,
  TANH = 266,
  MM = 333,
  CHUNK = 400
};

struct Entry {
  HashKey key;
  void * value;
};
constexpr int table_size = 38431;
Entry* table;


std::ostream& operator<<(std::ostream& out, const HashKey& key) {
  return out << "{" << key.inside[0] << ", " << key.inside[1] << ", " << key.inside[2] << ", " << key.inside[3] << "}";
}

void * lookup(const HashKey& key) {
  size_t start = key.hash() % table_size;
  // std::cout << key << " " << key.hash() << " " << start << "\n";
  while(!(table[start].key == key)) {
    JIT_ASSERT(!table[start].key.empty());
    start++;
  }
  // std::cout << "found at: " << start << "\n";
  return table[start].value;
}

// Ten is the base class for Tensor which handles the reference counting
struct Ten {
  Ten(): Ten(nullptr, false) {}
  Ten(TenImpl * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Ten(const Ten & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Ten(Ten && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Ten() {
    if (pImpl)
      pImpl->release();
  }
  Ten & operator=(Ten && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Ten & operator=(Ten const & rhs) & {
      //Ten ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Ten dtor releases rhs.pImpl, which was originally this->pImpl
      Ten(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Ten().swap(*this);
  }
  void reset(TenImpl * rhs) {
    Ten(rhs, true).swap(*this);
  }
  void reset(TenImpl * rhs, bool retain) {
    Ten(rhs, retain).swap(*this );
  }
  void swap(Ten & rhs) {
    TenImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TenImpl * get() const {
    return pImpl;
  }
  TenImpl * detach() {
    TenImpl * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  float* data() const {
    return pImpl->data;
  }
  int32_t key() const {
    return pImpl->hash_key;
  }
  bool defined() const {
    return pImpl;
  }
  #define LOOKUP(fn, ...) ((decltype(&fn))(lookup(HashKey(__VA_ARGS__))))
  Ten operator+(const Ten& b) const {
    return LOOKUP(plus_f, ADD, key(), b.key())(*this, b);
  }
  Ten operator*(const Ten& b) const {
    return LOOKUP(mul_f, MUL, key(), b.key())(*this, b);
  }
  Ten mm(const Ten& b) const {
    return LOOKUP(mm_f, MM, key(), b.key())(*this, b);
  }
  Ten sigmoid() const {
    return LOOKUP(sigmoid_f, SIGMOID, key())(*this);
  }
  Ten tanh() const {
    return LOOKUP(tanh_f, TANH, key())(*this);
  }
  std::vector<Ten> chunk4() {
    return LOOKUP(chunk4_f, CHUNK, key())(*this);
  }
  #undef LOOKUP
  at::ArrayRef<int64_t> sizes() const {
    return ArrayRef<int64_t>(pImpl->sizes, 2);
  }
public:
  TenImpl * pImpl;
};

Ten create(at::ArrayRef<int64_t> sizes) {
  auto r = Ten(new TenImpl, false);
  r.pImpl->hash_key = cpu_float_tensor_key;
  r.pImpl->sizes[0] = sizes[0];
  r.pImpl->sizes[1] = sizes[1];
  r.pImpl->data = new float[sizes[0]*sizes[1]];
  return r;
}

Ten plus_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] + b_data[i];
  }
  return c;
}

Ten mul_f(const Ten& a, const Ten& b) {
  JIT_ASSERT(a.sizes().equals(b.sizes()));
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i] * b_data[i];
  }
  return c;
}

Ten sigmoid_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = 1.f / (1.f + expf(-a_data[i]));
  }
  return c;
}

Ten tanh_f(const Ten& a) {
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = tanhf(a_data[i]);
  }
  return c;
}

Ten mm_f(const Ten& a, const Ten& b) {
  auto M = a.sizes()[0];
  auto K = a.sizes()[1];
  JIT_ASSERT(K == b.sizes()[0]);
  auto N = b.sizes()[1];
  auto c = create({M, N});
  auto a_data = a.data();
  auto b_data = b.data();
  auto c_data = c.data();

  for(int64_t m = 0; m < M; m++) {
    for(int64_t n = 0; n < N; n++) {
      float s = 0;
      for(int64_t k = 0; k < K; k++) {
        s += a_data[m*K + k] * b_data[k*N + n];
      }
      c_data[m*N + n] = s;
    }
  }
  return c;
}

std::vector<Ten> chunk4_f(const Ten& a) {
  JIT_ASSERT(a.sizes()[1] % 4 == 0);
  auto B = a.sizes()[0];
  auto M = a.sizes()[1];
  auto N = M / 4;
  std::vector<Ten> result = { create({B, N}), create({B, N}), create({B, N}), create({B, N}) };
  auto a_data = a.data();
  for(int64_t b = 0; b < B; b++) {
    for(int64_t i = 0; i < 4; i++) {
      for(int64_t n = 0; n < N; n++) {
        result[i].data()[b*N + n] = a_data[b*M + i*N + n];
      }
    }
  }
  return result;
}

Ten load(const at::Tensor& a) {
  JIT_ASSERT(a.is_contiguous());
  auto c = create(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data<float>();
  auto c_data = c.data();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}


at::Tensor save(const Ten& a) {
  auto c = at::CPU(at::kFloat).tensor(a.sizes());
  auto sizes = a.sizes();
  auto a_data = a.data();
  auto c_data = c.data<float>();
  int64_t N = sizes[0]*sizes[1];
  for(int64_t i = 0; i < N; i++) {
    c_data[i] = a_data[i];
  }
  return c;
}

std::pair<Ten, Ten>
lstm_t(Ten input,
      Ten hx,
      Ten cx,
      Ten w_ih,
      Ten w_hh) {
  auto gates = input.mm(w_ih) + hx.mm(w_hh);

  auto chunked_gates = gates.chunk4();
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

void noop() {
  std::cout << "NOP\n";
}


void insert(const HashKey& key, void* value) {
  size_t start = key.hash() % table_size;
  //std::cout << key << " " << key.hash() << " " << start << "\n";
  for(size_t i = 0;!table[start].key.empty() && !(table[start].key == key); i++, start++) {
    //std::cout << "collide: " << i << " " << table[start].key << " " << (table[start].key == key) << "\n";
  }
  table[start].key = key;
  table[start].value = value;
  //std::cout << "inserted: " << start << "\n";

}

void doit() {
  table = new Entry[table_size];
  // make up some operators
  for(int scalar_type = 0; scalar_type < 8; scalar_type++) {
    for(int backend = 0; backend < 4; backend++) {
      int32_t hash_key = scalar_type*4 + backend;
      for(int op = 1; op < 402; op++) {
        if(op % 2 == 0) {
          insert(HashKey(op, hash_key), (void*)noop);
        } else {
          insert(HashKey(op, hash_key, hash_key), (void*)noop);
        }
      }
    }
  }
  // register the real ones...

  insert(HashKey(ADD,cpu_float_tensor_key,cpu_float_tensor_key), (void*) plus_f);
  insert(HashKey(MUL,cpu_float_tensor_key,cpu_float_tensor_key), (void*)mul_f);
  insert(HashKey(MM,cpu_float_tensor_key,cpu_float_tensor_key), (void*)mm_f);
  insert(HashKey(SIGMOID,cpu_float_tensor_key),  (void*)sigmoid_f);
  insert(HashKey(TANH,cpu_float_tensor_key),  (void*)tanh_f);
  insert(HashKey(CHUNK,cpu_float_tensor_key),  (void*)chunk4_f);

  JIT_ASSERT(lookup(HashKey(ADD, cpu_float_tensor_key, cpu_float_tensor_key)) == (void*) plus_f);
  JIT_ASSERT(lookup(HashKey(MUL, cpu_float_tensor_key, cpu_float_tensor_key)) == (void*) mul_f);
  JIT_ASSERT(lookup(HashKey(MM, cpu_float_tensor_key, cpu_float_tensor_key)) == (void*)  mm_f);
  JIT_ASSERT(lookup(HashKey(SIGMOID, cpu_float_tensor_key)) == (void*)  sigmoid_f);
  JIT_ASSERT(lookup(HashKey(TANH, cpu_float_tensor_key)) == (void*)  tanh_f);
  JIT_ASSERT(lookup(HashKey(CHUNK, cpu_float_tensor_key)) == (void*)  chunk4_f);


  auto input = at::randn(at::CPU(at::kFloat), {batch_size, input_size});
  auto hx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto cx    = at::randn(at::CPU(at::kFloat), {batch_size, hidden_size});
  auto w_ih  = at::randn(at::CPU(at::kFloat), {input_size, 4 * hidden_size});
  auto w_hh  = at::randn(at::CPU(at::kFloat), {hidden_size, 4 * hidden_size});

  auto r0 = torch::jit::lstm(input, hx, cx, w_ih, w_hh);

  auto input_t = load(input);
  auto hx_t    = load(hx);
  auto cx_t    = load(cx);
  auto w_ih_t  = load(w_ih);
  auto w_hh_t  = load(w_hh);

  auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);

  auto r0_0 = r0.first;
  auto r0_1 = r0.second;
  auto r1_0 = save(r1.first);
  auto r1_1 = save(r1.second);

  JIT_ASSERT(torch::jit::almostEqual(r0_0, r1_0));
  JIT_ASSERT(torch::jit::almostEqual(r0_1, r1_1));
  auto start = getTime();
  size_t i;
  constexpr auto repeat = 100;
  for(i = 0; (getTime() - start) < 1000ULL*1000ULL*100ULL; ++i) {
    for(int j = 0; j < repeat; j++) {
      auto r1 = lstm_t(input_t, hx_t, cx_t, w_ih_t, w_hh_t);
    }
  }

  std::cout << "customhash " << (getTime() - start)/ (double)(i*repeat) << "\n";
}

}



std::string runJITCPPTests() {
  std::stringstream out;
  cpu_pin(16);
  check_cpu_governor(16);
  for(int i = 0; i < 100; i++) {
    flushCache(1);
    direct::doit();
    flushCache(1);
    vtable::doit();
    flushCache(1);
    custom_vtable::doit();
    flushCache(1);
    slowhash::doit();
    flushCache(1);
    consthash::doit();
    flushCache(1);
    customhash::doit();
  }
  std::cout << "DONE\n";
  testControlFlow();
  testGraphExecutor();
  testBlocks(out);
  testCreateAutodiffSubgraphs(out);
  testDifferentiate(out);
  testDifferentiateWithRequiresGrad(out);
  testADFormulas();
  interpTest();
  interpStageTest();
  codeTemplateTest();
  fusionTests();
  attributesTest();
  internedStringsTests();
  fromQualStringTests();
  argumentSpecTest();
  shapeAnalysisTest();
  return out.str();
}

}}
