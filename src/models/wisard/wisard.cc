
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
    cfg.validate();
    checkInputSizes(images.size(), labels.size());
    checkInputSizes(valImages.size(), valLabels.size());
    if(images.size() == 0){
      throw Exception("The training set can't be empty!");
    }
    checkConfidence(calculateNumberOfRams(images[0].size(), addressSize, completeAddressing));

    const bool earlyStopping = valImages.size() > 0;
    EpochRNG rng(cfg.seed);
    EpochReport report;
    std::map<std::string, Discriminator> best;
    double bestScore = -1;
    int since = 0;
    int lastEpoch = 0;
    std::vector<size_t> order(images.size());
    std::iota(order.begin(), order.end(), 0);

    beginTrainingAll();
    try{
      for(int t=0; t<cfg.epochs; t++){
        lastEpoch = t;
        const int bT = cfg.b*(t+1);
        const double rhoT = cfg.rhoAt(t);
        EpochCounters counters;

        rng.shuffle(order);
        for(size_t start=0; start<order.size(); start+=cfg.batchSize){
          size_t end = std::min(order.size(), start+(size_t)cfg.batchSize);
          for(size_t k=start; k<end; k++){
            if(verbose) std::cout << "\repoch " << t+1 << ": training " << k+1 << " of " << order.size();
            epochStep(images[order[k]], labels[order[k]], bT, rhoT, cfg, rng, counters);
          }
        }
        if(verbose) std::cout << "\r" << std::endl;

        double accVal = 0;
        if(earlyStopping){
          accVal = cfg.valMode == VAL_BLEACHING
            ? evaluateDeliveredBleaching(valImages, valLabels, t+1)
            : evaluateDelivered(valImages, valLabels, cfg.b, t+1);
        }
        std::map<std::string, double> row;
        row["epoch"] = t;
        row["rho"] = rhoT;
        row["bT"] = bT;
        row["accTrain"] = counters.hits/(double)images.size();
        row["accVal"] = accVal;
        row["nPenError"] = counters.penError;
        row["nPenMargin"] = counters.penMargin;
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


protected:
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

  template<typename T>
  void epochStep(const T& image, const std::string& label, const int bT, const double rhoT, const EpochConfig& cfg, EpochRNG& rng, EpochCounters& counters){
    std::map<std::string,int> R = _responses<T>(image, bT);
    if(!R.empty()){
      std::string w, l;
      const bool hasL = top2(R, w, l);
      const double Rw = R[w];
      const double Rl = hasL ? R[l] : 0;
      auto itY = R.find(label);
      const double Ry = itY == R.end() ? 0 : itY->second;
      const double C = Rw > 0 ? (Rw-Rl)/Rw : 0;

      if(w != label){
        double rhoEf = rhoT;
        if(cfg.modulate) rhoEf *= Rw > 0 ? (Rw-Ry)/Rw : 0;
        if(penalizeSelected(w, image, cfg.selectError, 1-rhoEf, bT, cfg, rng)) counters.penError++;
      }
      else{
        counters.hits++;
        if(hasL && cfg.tau > 0 && C < cfg.tau){
          double rhoEf = rhoT*(cfg.tau-C)/cfg.tau;
          if(penalizeSelected(l, image, cfg.selectMargin, 1-rhoEf, bT, cfg, rng)) counters.penMargin++;
        }
      }
    }

    if(discriminators.find(label) == discriminators.end()){
      makeDiscriminator(label, image.size());
      discriminators[label].beginTraining();
    }
    discriminators[label].trainF(image);
  }

  // RF-09: all active addresses, or the kappa with the lowest counter among the stored
  // ones (lowest_count) or among the active ones (closest_to_b)
  template<typename T>
  bool penalizeSelected(const std::string& label, const T& image, const SelectMode mode, const double factor, const int bT, const EpochConfig& cfg, EpochRNG& rng){
    Discriminator& d = discriminators[label];
    std::vector<std::pair<int,addr_t>> addrs = d.addresses(image, mode != SELECT_LOWEST_COUNT, bT);
    if(mode != SELECT_ALL){
      const size_t kappa = cfg.kappa > 0 ? cfg.kappa : (d.getNumberOfRAMS()+3)/4;
      std::stable_sort(addrs.begin(), addrs.end(),
        [&d](const std::pair<int,addr_t>& lhs, const std::pair<int,addr_t>& rhs){
          return d.getCounter(lhs.first, lhs.second) < d.getCounter(rhs.first, rhs.second);
        });
      if(addrs.size() > kappa) addrs.resize(kappa);
    }
    for(unsigned int i=0; i<addrs.size(); i++){
      d.penalize(addrs[i].first, addrs[i].second, factor, cfg.counterMode, rng);
    }
    return !addrs.empty();
  }

  // accuracy of the model as it would be delivered after dividing the counters by divisor;
  // it never writes in the RAMs
  double evaluateDelivered(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels, const int b, const double divisor) const{
    long hits = 0;
    for(unsigned int i=0; i<images.size(); i++){
      std::map<std::string,int> R;
      for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
        R[d->first] = d->second.responseDelivered(images[i], b, divisor);
      }
      if(R.empty()) continue;
      std::string w, l;
      top2(R, w, l);
      if(w == labels[i]) hits++;
    }
    return hits/(double)images.size();
  }

  // the same as above, but deciding like classify: iterative bleaching and getBiggestCandidate
  double evaluateDeliveredBleaching(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels, const double divisor) const{
    long hits = 0;
    for(unsigned int i=0; i<images.size(); i++){
      std::map<std::string,std::vector<int>> allvotes;
      for(auto d=discriminators.begin(); d!=discriminators.end(); ++d){
        allvotes[d->first] = d->second.votesDelivered(images[i], divisor);
      }
      std::map<std::string,int> candidates = Bleaching::make(allvotes, bleachingActivated, searchBestConfidence, confidence);
      if(Bleaching::getBiggestCandidate(candidates) == labels[i]) hits++;
    }
    return hits/(double)images.size();
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
    //float numberOfRAMS = calculateNumberOfRams(images[0].size(), addressSize, completeAddressing);
    std::vector<std::string> labels(images.size());

    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rclassifying " << i+1 << " of " << images.size();
      std::map<std::string,int> candidates = classify(images[i],searchBestConfidence);
      labels[i] = Bleaching::getBiggestCandidate(candidates);
    }
    if(verbose) std::cout << "\r" << std::endl;
    return labels;
  }

  std::map<std::string, int> classify(const std::vector<int>& image, bool searchBestConfidence=false){
    return __classify<std::vector<int>>(image,searchBestConfidence);
  }

  std::map<std::string, int> classify(const BinInput& image, bool searchBestConfidence=false){
    return __classify<BinInput>(image,searchBestConfidence);
  }

  template<typename T>
  std::map<std::string, int> __classify(const T& image, bool searchBestConfidence=false){
    std::map<std::string,std::vector<int>> allvotes;

    for(std::map<std::string,Discriminator>::iterator i=discriminators.begin(); i!=discriminators.end(); ++i){
      allvotes[i->first] = i->second.classify(image);
    }
    return Bleaching::make(allvotes, bleachingActivated, searchBestConfidence, confidence);
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
      {"returnClassesDegrees", returnClassesDegrees}
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
  int confidence;
  std::map<std::string, Discriminator> discriminators;
};
