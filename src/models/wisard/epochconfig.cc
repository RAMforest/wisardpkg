
enum CounterMode { COUNTER_FLOAT, COUNTER_INT_STOCHASTIC };
enum SelectMode { SELECT_ALL, SELECT_LOWEST_COUNT, SELECT_CLOSEST_TO_B };
enum DecayMode { DECAY_CONSTANT, DECAY_INVERSE, DECAY_GEOMETRIC };
enum ValMode { VAL_FIXED, VAL_BLEACHING };

// Only the mt19937 engine is used: its output is fixed by the standard, while std::shuffle
// and the std::*_distribution classes change between compilers, so the same seed gives
// the same training in any platform.
class EpochRNG{
public:
  EpochRNG(unsigned int seed=0): engine(seed){}

  double uniform01(){
    return next() / 4294967296.0;
  }

  // unbiased integer in [0, n)
  unsigned long long below(unsigned long long n){
    const unsigned long long range = 4294967296ULL;
    const unsigned long long limit = range - (range % n);
    unsigned long long value;
    do{
      value = next();
    }while(value >= limit);
    return value % n;
  }

  // Fisher-Yates
  void shuffle(std::vector<size_t>& values){
    for(size_t i=values.size(); i>1; i--){
      size_t j = below(i);
      std::swap(values[i-1], values[j]);
    }
  }

private:
  std::mt19937 engine;

  unsigned long long next(){
    return (unsigned long long)engine() & 0xFFFFFFFFULL;
  }
};

class EpochConfig{
public:
  double rho0 = 0.25;
  double tau = 0.1;
  int b = 1;
  DecayMode decay = DECAY_INVERSE;
  double gamma = 0.8;
  bool modulate = false;
  SelectMode selectError = SELECT_ALL;
  SelectMode selectMargin = SELECT_CLOSEST_TO_B;
  int kappa = 0; // 0 means ceil(numberOfRAMS/4)
  CounterMode counterMode = COUNTER_FLOAT;
  int batchSize = 64;
  int epochs = 20;
  int patience = 3;
  unsigned int seed = 0;
  // validation of each epoch: at the fixed threshold b, or with the iterative bleaching of classify
  ValMode valMode = VAL_FIXED;

  EpochConfig(){}
  EpochConfig(nl::json c){
    for(nl::json::iterator it = c.begin(); it != c.end(); ++it){
      const std::string key = it.key();
      if(key == "rho0") rho0 = it.value().get<double>();
      else if(key == "tau") tau = it.value().get<double>();
      else if(key == "b") b = it.value().get<int>();
      else if(key == "decay") decay = parseDecay(it.value().get<std::string>());
      else if(key == "gamma") gamma = it.value().get<double>();
      else if(key == "modulate") modulate = it.value().get<bool>();
      else if(key == "selectError") selectError = parseSelect(it.value().get<std::string>());
      else if(key == "selectMargin") selectMargin = parseSelect(it.value().get<std::string>());
      else if(key == "kappa") kappa = it.value().get<int>();
      else if(key == "counterMode") counterMode = parseCounterMode(it.value().get<std::string>());
      else if(key == "batchSize") batchSize = it.value().get<int>();
      else if(key == "epochs") epochs = it.value().get<int>();
      else if(key == "patience") patience = it.value().get<int>();
      else if(key == "seed") seed = it.value().get<unsigned int>();
      else if(key == "valMode") valMode = parseValMode(it.value().get<std::string>());
      else throw Exception("Unknown option for epoch training!");
    }
  }

  void validate() const{
    if(rho0 < 0 || rho0 >= 1) throw Exception("rho0 must be in [0, 1)!");
    if(tau < 0 || tau >= 1) throw Exception("tau must be in [0, 1)!");
    if(b < 1) throw Exception("b can't be lesser than 1!");
    if(gamma <= 0 || gamma > 1) throw Exception("gamma must be in (0, 1]!");
    if(selectError == SELECT_CLOSEST_TO_B) throw Exception("selectError must be all or lowest_count!");
    if(kappa < 0) throw Exception("kappa can't be negative!");
    if(batchSize < 1) throw Exception("batchSize can't be lesser than 1!");
    if(epochs < 1) throw Exception("epochs can't be lesser than 1!");
    if(patience < 1) throw Exception("patience can't be lesser than 1!");
  }

  double rhoAt(const int t) const{
    switch(decay){
      case DECAY_CONSTANT: return rho0;
      case DECAY_INVERSE: return rho0/(1+t);
      default: return rho0*std::pow(gamma, t);
    }
  }

  static CounterMode parseCounterMode(const std::string& name){
    if(name == "float") return COUNTER_FLOAT;
    if(name == "int_stochastic") return COUNTER_INT_STOCHASTIC;
    throw Exception("counterMode must be float or int_stochastic!");
  }

  static SelectMode parseSelect(const std::string& name){
    if(name == "all") return SELECT_ALL;
    if(name == "lowest_count") return SELECT_LOWEST_COUNT;
    if(name == "closest_to_b") return SELECT_CLOSEST_TO_B;
    throw Exception("The address selection must be all, lowest_count or closest_to_b!");
  }

  static DecayMode parseDecay(const std::string& name){
    if(name == "constant") return DECAY_CONSTANT;
    if(name == "inverse") return DECAY_INVERSE;
    if(name == "geometric") return DECAY_GEOMETRIC;
    throw Exception("decay must be constant, inverse or geometric!");
  }

  static ValMode parseValMode(const std::string& name){
    if(name == "fixed") return VAL_FIXED;
    if(name == "bleaching") return VAL_BLEACHING;
    throw Exception("valMode must be fixed or bleaching!");
  }
};

class EpochReport{
public:
  int bestEpoch = -1;
  std::vector<std::map<std::string, double>> perEpoch;
};
