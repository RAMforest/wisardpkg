
class Wisard{
public:
  Wisard(int addressSize): Wisard(addressSize, {}){}
  Wisard(int addressSize, nl::json c): addressSize(addressSize){
    srand(randint(0,1000000));
    nl::json value;

    value = c["bleachingActivated"];
    bleachingActivated = value.is_null() ? true : value.get<bool>();

    value = c["verbose"];
    verbose = value.is_null() ? false : value.get<bool>();

    value = c["ignoreZero"];
    ignoreZero = value.is_null() ? false : value.get<bool>();

    value = c["completeAddressing"];
    completeAddressing = value.is_null() ? true : value.get<bool>();

    value = c["mapping"];
    mapping = value.is_null() ? std::map<std::string, std::vector<std::vector<int>>>() : value.get<std::map<std::string, std::vector<std::vector<int>>>>();

    value = c["indexes"];
    indexes = value.is_null() ? std::vector<int>(0) : value.get<std::vector<int>>();

    value = c["base"];
    base = value.is_null() ? 2 : value.get<int>();

    value = c["confidence"];
    confidence = value.is_null() ? 1 : value.get<int>();

    value = c["searchBestConfidence"];
    searchBestConfidence = value.is_null() ? false : value.get<bool>();

    value = c["returnConfidence"];
    returnConfidence = value.is_null() ? false : value.get<bool>();

    value = c["returnActivationDegree"];
    returnActivationDegree = value.is_null() ? false : value.get<bool>();

    value = c["returnClassesDegrees"];
    returnClassesDegrees = value.is_null() ? false : value.get<bool>();

    value = c["returnBleachingInfo"];
    returnBleachingInfo = value.is_null() ? false : value.get<bool>();

    value = c["bleaching"];
    bleachingMode = value.is_null() ? "iterative" : value.get<std::string>();

    value = c["confidenceTolerance"];
    confidenceTolerance = value.is_null() ? 0.01 : value.get<double>();

    value = c["freqTolerance"];
    freqTolerance = value.is_null() ? 0.1 : value.get<double>();

    value = c["dabRemoval"];
    dabRemoval = value.is_null() ? "inclusive" : value.get<std::string>();

    value = c["multiBleaching"];
    multiBleaching = value.is_null() ? false : value.get<bool>();

    value = c["confidenceThreshold"];
    confidenceThreshold = value.is_null() ? 0.1 : value.get<double>();

    value = c["tieTolerance"];
    tieTolerance = value.is_null() ? 0.0 : value.get<double>();

    value = c["nextCut"];
    nextCut = value.is_null() ? "ref" : value.get<std::string>();

    value = c["maxRounds"];
    maxRounds = value.is_null() ? 64 : value.get<int>();

    checkBleachingOptions();
  }

  Wisard(std::string config):Wisard(0,nl::json::parse(config)){
    nl::json c = nl::json::parse(config);
    addressSize=c["addressSize"];

    nl::json classes = c["classes"];
    nl::json dConfig = {
      {"ignoreZero", ignoreZero},
      {"base", base}
    };
    for(nl::json::iterator it = classes.begin(); it != classes.end(); ++it){
      nl::json d = it.value();
      d.merge_patch(dConfig);
      discriminators[it.key()] = Discriminator(d);
    }
  }

  long getsizeof(){
    long size = sizeof(Wisard);
    size += sizeof(int)*indexes.size();
    for(std::map<std::string, Discriminator>::iterator d=discriminators.begin(); d!=discriminators.end(); ++d){
      size += d->first.size() + d->second.getsizeof();
    }
    return size;
  }

  ~Wisard(){
    indexes.clear();
    discriminators.clear();
  }

  void train(const DataSet& dataset) {
    int numberOfRAMS = calculateNumberOfRams(dataset[0].size(), addressSize, completeAddressing);
    checkConfidence(numberOfRAMS);
    for(size_t i=0; i<dataset.size(); i++){
      if(verbose) std::cout << "\rtraining " << i+1 << " of " << dataset.size();
      train<BinInput>(dataset[i], dataset.getLabel(i));
    }
  }

  void train(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels){
    int numberOfRAMS = calculateNumberOfRams(images[0].size(), addressSize, completeAddressing);
    checkConfidence(numberOfRAMS);
    checkInputSizes(images.size(), labels.size());
    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rtraining " << i+1 << " of " << images.size();
      train<std::vector<int>>(images[i], labels[i]);
    }
    if(verbose) std::cout << "\r" << std::endl;
  }

  std::vector<std::string> classify(const std::vector<std::vector<int>>& images){
    return _classify<std::vector<std::vector<int>>>(images);
  }

  std::vector<std::string> classify(const DataSet& images){
    return _classify<DataSet>(images);
  }

  void leaveOneOut(const std::vector<int>& image, const std::string& label){
    auto d = discriminators.find(label);
    if(d != discriminators.end()){
      d->second.untrain(image);
    }
  }

  void leaveMoreOut(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels){
    checkInputSizes(images.size(), labels.size());
    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\runtraining " << i+1 << " of " << images.size();
      leaveOneOut(images[i], labels[i]);
    }
    if(verbose) std::cout << "\r" << std::endl;
  }

  std::map<std::string,std::vector<int>> getMentalImages(){
    std::map<std::string,std::vector<int>> images;
    for(std::map<std::string, Discriminator>::iterator d=discriminators.begin(); d!=discriminators.end(); ++d){
      images[d->first] = d->second.getMentalImage();
    }
    return images;
  }

  std::string jsonConfig(){
    nl::json config = getConfig();
    config["classes"] = getConfigClassesJSON();
    return config.dump(2);
  }

  std::string json(bool huge, std::string path) {
    nl::json config = getConfig();
    config["classes"] = getClassesJSON(huge,path);
    return config.dump();
  }
  std::string json(bool huge) {
    return json(huge,"");
  }
  std::string json() {
    return json(false,"");
  }

  // the counters classify reads in every RAM of every discriminator, before any bleaching
  std::map<std::string,std::vector<int>> votes(const std::vector<int>& image){
    return _votes<std::vector<int>>(image);
  }

  std::map<std::string,std::vector<int>> votes(const BinInput& image){
    return _votes<BinInput>(image);
  }

  // R_d(x) for every discriminator at a fixed threshold b, without bleaching
  std::map<std::string,int> responses(const std::vector<int>& image, const int b) const{
    return _responses<std::vector<int>>(image, b);
  }

  std::map<std::string,int> responses(const BinInput& image, const int b) const{
    return _responses<BinInput>(image, b);
  }

  // (ram, address) of image on the discriminator of label with counter >= b
  std::vector<std::pair<int,addr_t>> activeAddresses(const std::vector<int>& image, const std::string& label, const int b) const{
    return getDiscriminator(label).addresses(image, true, b);
  }

  void penalize(const std::string& label, const std::vector<std::pair<int,addr_t>>& addrs, const double factor, const CounterMode mode=COUNTER_FLOAT, const unsigned int seed=0){
    if(factor < 0 || factor > 1){
      throw Exception("The penalization factor must be in [0, 1]!");
    }
    Discriminator& d = getDiscriminator(label);
    EpochRNG rng(seed);
    for(unsigned int i=0; i<addrs.size(); i++){
      d.penalize(addrs[i].first, addrs[i].second, factor, mode, rng);
    }
  }

  // Epoch training (requisito-implementação.md, section 5): each example is classified at a
  // fixed threshold before being written, the wrong winner (or the runner-up when the margin
  // is low) is penalized multiplicatively and the right class is reinforced. The model ends
  // at the best validation epoch, with counters divided by the number of epochs it has seen.
  EpochReport trainEpochs(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels,
                          const std::vector<std::vector<int>>& valImages, const std::vector<std::string>& valLabels,
                          const EpochConfig& cfg){
    checkInputSizes(images.size(), labels.size());
    checkInputSizes(valImages.size(), valLabels.size());
    return _trainEpochs(VectorSamples(images, labels), VectorSamples(valImages, valLabels), cfg);
  }

  // the DataSets must have a label in every example
  EpochReport trainEpochs(const DataSet& train, const DataSet& val, const EpochConfig& cfg){
    return _trainEpochs(DataSetSamples(train), DataSetSamples(val), cfg);
  }


protected:
  struct VectorSamples{
    const std::vector<std::vector<int>>& images;
    const std::vector<std::string>& labels;
    VectorSamples(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels): images(images), labels(labels){}
    size_t size() const{ return images.size(); }
    const std::vector<int>& image(size_t i) const{ return images[i]; }
    const std::string& label(size_t i) const{ return labels[i]; }
  };

  struct DataSetSamples{
    const DataSet& data;
    DataSetSamples(const DataSet& data): data(data){}
    size_t size() const{ return data.size(); }
    const BinInput& image(size_t i) const{ return data[i]; }
    std::string label(size_t i) const{ return data.getLabel(i); }
  };

  // Memory for the addresses computed before the first epoch (AddressTable), training and
  // validation together; past it, the addresses of a discriminator are computed at each use.
  static const size_t epochAddressBudget = (size_t)1 << 30;

  template<typename S>
  EpochReport _trainEpochs(const S& train, const S& val, const EpochConfig& cfg){
    cfg.validate();
    if(train.size() == 0){
      throw Exception("The training set can't be empty!");
    }
    checkConfidence(calculateNumberOfRams(train.image(0).size(), addressSize, completeAddressing));

    const bool earlyStopping = val.size() > 0;
    const int threads = epochThreads(cfg.threads);
    EpochRNG rng(cfg.seed);
    EpochReport report;
    std::map<std::string, Discriminator> best;
    double bestScore = -1;
    int since = 0;
    int lastEpoch = 0;
    std::vector<size_t> order(train.size());
    std::iota(order.begin(), order.end(), 0);

    // the classes, in the order of discriminators, and the class of every training example
    std::vector<int> trainClass(train.size());
    std::vector<EpochClass> classes = epochClasses(train, trainClass);
    EpochTables trainTables, valTables;
    size_t budget = epochAddressBudget;

    beginTrainingAll();
    try{
      // the addresses computed before the first epoch count in its training time
      auto trainStart = std::chrono::steady_clock::now();
      for(size_t c=0; c<classes.size(); c++){
        if(classes[c].d != nullptr) attachClass(classes[c], *classes[c].d, train, trainTables, threads, budget);
      }
      for(int t=0; t<cfg.epochs; t++){
        lastEpoch = t;
        const int bT = cfg.b*(t+1);
        const double rhoT = cfg.rhoAt(t);
        EpochCounters counters;
        if(t > 0) trainStart = std::chrono::steady_clock::now();

        rng.shuffle(order);
        for(size_t start=0; start<order.size(); start+=cfg.batchSize){
          size_t end = std::min(order.size(), start+(size_t)cfg.batchSize);
          for(size_t k=start; k<end; k++){
            if(verbose) std::cout << "\repoch " << t+1 << ": training " << k+1 << " of " << order.size();
            epochStep(train, order[k], classes, trainClass[order[k]], bT, rhoT, cfg, rng, counters,
                      trainTables, threads, budget);
          }
        }
        if(verbose) std::cout << "\r" << std::endl;
        const auto valStart = std::chrono::steady_clock::now();

        double accVal = 0;
        if(earlyStopping){
          accVal = evaluateEpoch(val, classes, valTables, cfg, t+1, threads, budget);
        }
        const auto valEnd = std::chrono::steady_clock::now();
        std::map<std::string, double> row;
        row["epoch"] = t;
        row["rho"] = rhoT;
        row["bT"] = bT;
        row["accTrain"] = counters.hits/(double)train.size();
        row["accVal"] = accVal;
        row["nPenError"] = counters.penError;
        row["nPenMargin"] = counters.penMargin;
        // wall time of the epoch: the pass over the training set and the validation
        row["secondsTrain"] = std::chrono::duration<double>(valStart - trainStart).count();
        row["secondsVal"] = std::chrono::duration<double>(valEnd - valStart).count();
        report.perEpoch.push_back(row);

        if(!earlyStopping) continue;
        if(accVal > bestScore){
          bestScore = accVal;
          best = discriminators;
          report.bestEpoch = t;
          since = 0;
        }
        else if(++since >= cfg.patience){
          break;
        }
      }

      if(earlyStopping){
        discriminators = best;
      }
      else{
        report.bestEpoch = lastEpoch;
      }
      endTrainingAll(report.bestEpoch+1);
    }
    catch(...){
      endTrainingAll(1);
      throw;
    }
    return report;
  }

  struct EpochCounters{
    long hits = 0;
    long penError = 0;
    long penMargin = 0;
  };

  template<typename T>
  std::map<std::string,int> _responses(const T& image, const int b) const{
    std::map<std::string,int> out;
    for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
      out[d->first] = d->second.response(image, b);
    }
    return out;
  }

  Discriminator& getDiscriminator(const std::string& label){
    auto d = discriminators.find(label);
    if(d == discriminators.end()){
      throw Exception("There is no discriminator for the given label!");
    }
    return d->second;
  }

  const Discriminator& getDiscriminator(const std::string& label) const{
    auto d = discriminators.find(label);
    if(d == discriminators.end()){
      throw Exception("There is no discriminator for the given label!");
    }
    return d->second;
  }

  // winner and runner-up; on ties the smallest label wins (std::map order)
  static bool top2(const std::map<std::string,int>& R, std::string& w, std::string& l){
    int bw = -1, bl = -1;
    bool hasL = false;
    for(auto it=R.begin(); it!=R.end(); ++it){
      if(it->second > bw){
        if(bw >= 0){
          l = w; bl = bw; hasL = true;
        }
        w = it->first; bw = it->second;
      }
      else if(it->second > bl){
        l = it->first; bl = it->second; hasL = true;
      }
    }
    return hasL;
  }

  // One class of the epoch training. Classes are indexes in the order of discriminators
  // (std::map), so ties still go to the smallest label.
  struct EpochClass{
    std::string label;
    Discriminator* d = nullptr;                 // null until the first example of the class
    std::shared_ptr<AddressTable> train, val;   // addresses of the examples; null: computed at each use
    bool valReady = false;
    std::vector<addr_t> addrs;                  // the addresses of the current example
    std::vector<counter_t*> counters;           // and their counters (RAM::findCounter)
    std::vector<int> selected;                  // RAMs chosen for a penalization
    int response = 0;
  };

  // address tables already built, shared by the discriminators with the same mapping
  typedef std::vector<std::pair<const Discriminator*, std::shared_ptr<AddressTable>>> EpochTables;

  template<typename S>
  std::vector<EpochClass> epochClasses(const S& train, std::vector<int>& trainClass){
    std::map<std::string,int> index;
    for(auto d=discriminators.begin(); d!=discriminators.end(); ++d) index[d->first] = 0;
    std::vector<std::string> labels(train.size());
    for(size_t i=0; i<train.size(); i++){
      labels[i] = train.label(i);
      index[labels[i]] = 0;
    }
    std::vector<EpochClass> classes(index.size());
    int k = 0;
    for(auto it=index.begin(); it!=index.end(); ++it){
      it->second = k;
      classes[k].label = it->first;
      auto d = discriminators.find(it->first);
      if(d != discriminators.end()) classes[k].d = &d->second;
      k++;
    }
    for(size_t i=0; i<train.size(); i++) trainClass[i] = index[labels[i]];
    return classes;
  }

  template<typename S>
  std::shared_ptr<AddressTable> addressTable(const S& samples, const Discriminator& d, EpochTables& tables, const int threads, size_t& budget){
    for(size_t k=0; k<tables.size(); k++){
      if(tables[k].first->sameMapping(d)) return tables[k].second;
    }
    std::shared_ptr<AddressTable> table = AddressTable::build(samples, d, threads, budget);
    tables.push_back(std::make_pair(&d, table));
    return table;
  }

  template<typename S>
  void attachClass(EpochClass& e, Discriminator& d, const S& train, EpochTables& tables, const int threads, size_t& budget){
    e.d = &d;
    const size_t rams = d.getNumberOfRAMS();
    e.addrs.assign(rams, 0);
    e.counters.assign(rams, nullptr);
    e.selected.reserve(rams);
    e.train = addressTable(train, d, tables, threads, budget);
  }

  template<typename S>
  static void loadAddresses(const S& samples, const size_t i, const std::shared_ptr<AddressTable>& table, const Discriminator& d, addr_t* out){
    if(table) table->row(i, out);
    else d.fillAddresses(samples.image(i), out);
  }

  // One example of the epoch training: classified at the fixed threshold bT before being
  // written, the wrong winner (or the runner-up when the margin is low) is penalized
  // multiplicatively and the right class is reinforced. The address and the counter of every
  // RAM are read once and kept for the penalization and the reinforcement: the penalized
  // class is never the reinforced one, so the counters read stay valid.
  template<typename S>
  void epochStep(const S& train, const size_t i, std::vector<EpochClass>& classes, const int y, const int bT,
                 const double rhoT, const EpochConfig& cfg, EpochRNG& rng, EpochCounters& counters,
                 EpochTables& tables, const int threads, size_t& budget){
    // R_k: RAMs with counter >= bT; winner and runner-up, ties to the smallest label
    int w = -1, l = -1, bw = -1, bl = -1;
    bool hasL = false;
    for(size_t c=0; c<classes.size(); c++){
      EpochClass& e = classes[c];
      if(e.d == nullptr) continue;
      loadAddresses(train, i, e.train, *e.d, e.addrs.data());
      e.d->findCounters(e.addrs.data(), e.counters.data());
      int r = 0;
      for(size_t k=0; k<e.counters.size(); k++){
        if(e.counters[k] != nullptr && *e.counters[k] >= bT) r++;
      }
      e.response = r;
      if(r > bw){
        if(bw >= 0){
          l = w; bl = bw; hasL = true;
        }
        w = (int)c; bw = r;
      }
      else if(r > bl){
        l = (int)c; bl = r; hasL = true;
      }
    }

    if(w >= 0){
      const double Rw = bw;
      const double Rl = hasL ? bl : 0;
      const double Ry = classes[y].d != nullptr ? classes[y].response : 0;
      const double C = Rw > 0 ? (Rw-Rl)/Rw : 0;

      if(w != y){
        double rhoEf = rhoT;
        if(cfg.modulate) rhoEf *= Rw > 0 ? (Rw-Ry)/Rw : 0;
        if(penalizeSelected(classes[w], cfg.selectError, 1-rhoEf, bT, cfg, rng)) counters.penError++;
      }
      else{
        counters.hits++;
        if(hasL && cfg.tau > 0 && C < cfg.tau){
          double rhoEf = rhoT*(cfg.tau-C)/cfg.tau;
          if(penalizeSelected(classes[l], cfg.selectMargin, 1-rhoEf, bT, cfg, rng)) counters.penMargin++;
        }
      }
    }

    EpochClass& e = classes[y];
    if(e.d == nullptr){
      makeDiscriminator(e.label, train.image(i).size());
      Discriminator& d = discriminators[e.label];
      d.beginTraining();
      attachClass(e, d, train, tables, threads, budget);
      loadAddresses(train, i, e.train, d, e.addrs.data());
      d.findCounters(e.addrs.data(), e.counters.data());
    }
    e.d->trainAt(e.addrs.data(), e.counters.data());
  }

  // RF-09: all active addresses, or the kappa with the lowest counter among the stored
  // ones (lowest_count) or among the active ones (closest_to_b); e holds the counters read
  // by epochStep for the current example
  bool penalizeSelected(EpochClass& e, const SelectMode mode, const double factor, const int bT, const EpochConfig& cfg, EpochRNG& rng){
    std::vector<int>& selected = e.selected;
    selected.clear();
    for(size_t r=0; r<e.counters.size(); r++){
      const counter_t* counter = e.counters[r];
      if(counter != nullptr && (mode == SELECT_LOWEST_COUNT || *counter >= bT)) selected.push_back((int)r);
    }
    if(mode != SELECT_ALL){
      const size_t kappa = cfg.kappa > 0 ? cfg.kappa : (e.d->getNumberOfRAMS()+3)/4;
      std::stable_sort(selected.begin(), selected.end(), [&e](const int lhs, const int rhs){
        return *e.counters[lhs] < *e.counters[rhs];
      });
      if(selected.size() > kappa) selected.resize(kappa);
    }
    for(size_t k=0; k<selected.size(); k++){
      const int r = selected[k];
      e.d->penalizeAt(r, e.addrs[r], e.counters[r], factor, cfg.counterMode, rng);
    }
    return !selected.empty();
  }

  // Accuracy on samples of the model as it would be delivered after dividing the counters by
  // divisor: at the fixed threshold b (VAL_FIXED) or deciding like classify, with the bleaching
  // policy of the model (VAL_BLEACHING; the class sizes are the counts of the delivered model).
  // It never writes in the RAMs, so the examples are split among the threads.
  template<typename S>
  double evaluateEpoch(const S& samples, std::vector<EpochClass>& classes, EpochTables& tables,
                       const EpochConfig& cfg, const double divisor, const int threads, size_t& budget){
    std::vector<const EpochClass*> active;
    size_t rams = 0;
    for(size_t c=0; c<classes.size(); c++){
      EpochClass& e = classes[c];
      if(e.d == nullptr) continue;
      if(!e.valReady){
        e.val = addressTable(samples, *e.d, tables, threads, budget);
        e.valReady = true;
      }
      active.push_back(&e);
      rams = std::max(rams, e.addrs.size());
    }
    std::map<std::string,int> sizes;
    for(size_t c=0; c<active.size(); c++){
      sizes[active[c]->label] = (int)std::lround(active[c]->d->getNumberOfTrainings()/divisor);
    }
    const bool bleaching = cfg.valMode == VAL_BLEACHING;
    std::vector<long> hits(std::max(threads, 1), 0);
    parallelFor(samples.size(), threads, [&](size_t begin, size_t end, size_t part){
      std::unique_ptr<BleachingPolicy> policy = makeBleachingPolicy();
      std::vector<addr_t> addrs(rams);
      long h = 0;
      for(size_t i=begin; i<end; i++){
        const std::string& label = samples.label(i);
        if(bleaching){
          std::map<std::string,std::vector<int>> allvotes;
          for(size_t c=0; c<active.size(); c++){
            loadAddresses(samples, i, active[c]->val, *active[c]->d, addrs.data());
            active[c]->d->votesDeliveredAt(addrs.data(), divisor, allvotes[active[c]->label]);
          }
          if(policy->resolve(allvotes, sizes).label == label) h++;
        }
        else{
          int w = -1, bw = -1;
          for(size_t c=0; c<active.size(); c++){
            loadAddresses(samples, i, active[c]->val, *active[c]->d, addrs.data());
            const int r = active[c]->d->responseDeliveredAt(addrs.data(), cfg.b, divisor);
            if(r > bw){
              w = (int)c; bw = r;
            }
          }
          if(w >= 0 && active[w]->label == label) h++;
        }
      }
      hits[part] += h;
    });
    long total = 0;
    for(size_t p=0; p<hits.size(); p++) total += hits[p];
    return total/(double)samples.size();
  }

  void beginTrainingAll(){
    for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
      d->second.beginTraining();
    }
  }

  void endTrainingAll(const double divisor){
    for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
      d->second.endTraining(divisor);
    }
  }

  template<typename T>
  void train(const T& image, const std::string& label){
    if(discriminators.find(label) == discriminators.end()){
      makeDiscriminator(label, image.size());
    }
    discriminators[label].train(image);
  }

  template<typename T>
  std::vector<std::string> _classify(const T& images){
    std::vector<std::string> labels(images.size());
    std::unique_ptr<BleachingPolicy> policy = makeBleachingPolicy();
    const std::map<std::string,int> sizes = classSizes();

    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rclassifying " << i+1 << " of " << images.size();
      labels[i] = decide(images[i], *policy, sizes).label;
    }
    if(verbose) std::cout << "\r" << std::endl;
    return labels;
  }

  template<typename T>
  std::map<std::string,std::vector<int>> _votes(const T& image){
    std::map<std::string,std::vector<int>> allvotes;
    for(std::map<std::string,Discriminator>::iterator i=discriminators.begin(); i!=discriminators.end(); ++i){
      allvotes[i->first] = i->second.classify(image);
    }
    return allvotes;
  }

  template<typename T>
  BleachingResult decide(const T& image, const BleachingPolicy& policy, const std::map<std::string,int>& sizes){
    std::map<std::string,std::vector<int>> allvotes = _votes<T>(image);
    return policy.resolve(allvotes, sizes);
  }

  // with bleachingActivated=false there is no bleaching, whatever the policy
  std::unique_ptr<BleachingPolicy> makeBleachingPolicy() const{
    std::unique_ptr<BleachingPolicy> base = makeBasePolicy();
    if(bleachingActivated && multiBleaching){
      return std::unique_ptr<BleachingPolicy>(new MultiBleaching(std::move(base), confidenceThreshold, maxRounds));
    }
    return base;
  }

  std::unique_ptr<BleachingPolicy> makeBasePolicy() const{
    const bool inclusive = dabRemoval == "inclusive";
    if(bleachingActivated && bleachingMode == "dab"){
      return std::unique_ptr<BleachingPolicy>(new DAB(confidenceTolerance, inclusive));
    }
    if(bleachingActivated && bleachingMode == "rdab"){
      return std::unique_ptr<BleachingPolicy>(new RDAB(confidenceTolerance, freqTolerance, inclusive));
    }
    if(bleachingActivated && bleachingMode == "bbleaching"){
      return std::unique_ptr<BleachingPolicy>(new BBleachingPolicy(tieTolerance, nextCut == "ref"));
    }
    return std::unique_ptr<BleachingPolicy>(new IterativeBleaching(bleachingActivated, confidence));
  }

  // training examples of every class: the count of its discriminator, which also ends equal to
  // the class size after trainEpochs (every example is reinforced once per epoch)
  std::map<std::string,int> classSizes() const{
    std::map<std::string,int> sizes;
    for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
      sizes[d->first] = d->second.getNumberOfTrainings();
    }
    return sizes;
  }

  void checkBleachingOptions() const{
    if(bleachingMode != "iterative" && bleachingMode != "dab" && bleachingMode != "rdab" && bleachingMode != "bbleaching"){
      throw Exception("The bleaching must be iterative, dab, rdab or bbleaching!");
    }
    if(multiBleaching && bleachingMode == "rdab"){
      throw Exception("The multiBleaching works over iterative, dab or bbleaching, not rdab!");
    }
    if(nextCut != "ref" && nextCut != "untie"){
      throw Exception("The nextCut must be ref or untie!");
    }
    if(confidenceThreshold < 0 || confidenceThreshold > 1){
      throw Exception("The confidenceThreshold must be in [0, 1]!");
    }
    if(tieTolerance < 0 || tieTolerance >= 1){
      throw Exception("The tieTolerance must be in [0, 1)!");
    }
    if(maxRounds < 1){
      throw Exception("The maxRounds must be at least 1!");
    }
    if(dabRemoval != "inclusive" && dabRemoval != "exclusive"){
      throw Exception("The dabRemoval must be inclusive or exclusive!");
    }
    if(confidenceTolerance < 0 || confidenceTolerance > 1){
      throw Exception("The confidenceTolerance must be in [0, 1]!");
    }
    if(freqTolerance < 0 || freqTolerance > 1){
      throw Exception("The freqTolerance must be in [0, 1]!");
    }
  }

  nl::json getClassesJSON(bool huge, std::string path){
    nl::json c;
    for(std::map<std::string, Discriminator>::iterator d=discriminators.begin(); d!=discriminators.end(); ++d){
      c[d->first] = d->second.getJSON(huge,path+(d->first)+"__");
    }
    return c;
  }

  nl::json getConfigClassesJSON(){
    nl::json c;
    for(std::map<std::string, Discriminator>::iterator d=discriminators.begin(); d!=discriminators.end(); ++d){
      c[d->first] = d->second.getConfigJSON();
    }
    return c;
  }

  nl::json getConfig(){
    nl::json config = {
      {"version", __version__},
      {"addressSize", addressSize},
      {"bleachingActivated", bleachingActivated},
      {"verbose", verbose},
      {"indexes", indexes},
      {"ignoreZero", ignoreZero},
      {"completeAddressing", completeAddressing},
      {"base", base},
      {"confidence", confidence},
      // {"searchBestConfidence", searchBestConfidence},
      {"returnConfidence", returnConfidence},
      {"returnActivationDegree", returnActivationDegree},
      {"returnClassesDegrees", returnClassesDegrees},
      {"returnBleachingInfo", returnBleachingInfo},
      {"bleaching", bleachingMode},
      {"confidenceTolerance", confidenceTolerance},
      {"freqTolerance", freqTolerance},
      {"dabRemoval", dabRemoval},
      {"multiBleaching", multiBleaching},
      {"confidenceThreshold", confidenceThreshold},
      {"tieTolerance", tieTolerance},
      {"nextCut", nextCut},
      {"maxRounds", maxRounds}
    };
    return config;
  }

  void makeDiscriminator(std::string label, int entrySize){
    auto it = mapping.find(label);
    if (it != mapping.end())
    {
      discriminators[label] = Discriminator(it->second, entrySize, ignoreZero, base);
    }
    else if (indexes.size() != 0)
    {
      discriminators[label] = Discriminator(indexes, addressSize, entrySize, ignoreZero, base);
    }
    else
    {
      discriminators[label] = Discriminator(addressSize, entrySize, ignoreZero, completeAddressing, base);
    }
  }

  void checkInputSizes(const int imageSize, const int labelsSize){
    if(imageSize != labelsSize){
      throw Exception("The size of data is not the same of the size of labels!");
    }
  }

  void checkConfidence(int numberOfRAMS){
    if(confidence > numberOfRAMS){
      throw Exception("The confidence can not be bigger than number of RAMs!");
    }
  }

  int addressSize;
  bool bleachingActivated;
  bool verbose;
  std::map<std::string, std::vector<std::vector<int>>> mapping;
  std::vector<int> indexes;
  bool ignoreZero;
  bool completeAddressing;
  int base;
  bool searchBestConfidence;
  bool returnConfidence;
  bool returnActivationDegree;
  bool returnClassesDegrees;
  bool returnBleachingInfo;
  int confidence;
  std::string bleachingMode;  // iterative, dab, rdab or bbleaching (bleachingpolicy.cc)
  double confidenceTolerance;
  double freqTolerance;
  std::string dabRemoval;
  bool multiBleaching;         // sums the votes of the rounds of the base policy (MultiBleaching)
  double confidenceThreshold;  // multiBleaching: a round with confidence above it is the last
  double tieTolerance;         // bbleaching: confidence <= it is a tie
  std::string nextCut;         // bbleaching in the later rounds of multiBleaching: ref or untie
  int maxRounds;               // multiBleaching: most rounds
  std::map<std::string, Discriminator> discriminators;
};
