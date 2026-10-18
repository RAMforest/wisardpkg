
class RAM{
public:
  RAM(){}
  RAM(nl::json c){
    ignoreZero = c["ignoreZero"];
    base=c["base"];
    addresses = c["addresses"].get<std::vector<int>>();
    checkLimitAddressSize(addresses.size(), base);

    RAMDataHandle handle(c["data"].get<std::string>());
    positions = handle.get(0);
  }
  RAM(const int addressSize, const int entrySize, const bool ignoreZero=false, int base=2): ignoreZero(ignoreZero), base(base){
    checkLimitAddressSize(addressSize, base);
    addresses = std::vector<int>(addressSize);
    generateRandomAddresses(entrySize);
  }
  RAM(const std::vector<int> indexes, const bool ignoreZero=false, int base=2): addresses(indexes), ignoreZero(ignoreZero), base(base){
    checkLimitAddressSize(indexes.size(), base);
  }

  int getVote(const std::vector<int>& image){
    return getVote<std::vector<int>>(image);
  }

  int getVote(const BinInput& image){
    return getVote<BinInput>(image);
  }

  void train(const std::vector<int>& image){
    train<std::vector<int>>(image);
  }

  void train(const BinInput& image){
    train<BinInput>(image);
  }

  void untrain(const std::vector<int>& image){
      addr_t index = getIndex(image);
      auto it = positions.find(index);
      if(it != positions.end()){
        it->second--;
      }
  }

  std::vector<std::vector<int>> getMentalImage() {
    std::vector<std::vector<int>> mentalPiece(addresses.size());
    for(unsigned int i=0; i<mentalPiece.size(); i++){
      mentalPiece[i].resize(2);
      mentalPiece[i][0] = addresses[i];
      mentalPiece[i][1] = 0;
    }

    for(auto j=positions.begin(); j!=positions.end(); ++j){
      if(j->first == 0) continue;
      const std::vector<int> address = convertToBase(j->first);
      for(unsigned int i=0; i<mentalPiece.size(); i++){
        if(mentalPiece[i].size() == 0){
          mentalPiece[i].resize(2);
          mentalPiece[i][0] = addresses[i];
          mentalPiece[i][1] = 0;
        }
        if(address[i] > 0){
          mentalPiece[i][1] += j->second;
        }
      }
    }
    return mentalPiece;
  }

  nl::json getConfig(){
    nl::json config = {
      {"ignoreZero", ignoreZero},
      {"base", base}
    };
    return config;
  }

  std::string getData(){
    RAMDataHandle handle(positions);
    return handle.data(0);
  }

  void setMapping(std::vector<std::vector<int>>& mapping, int i){
    int size = addresses.size();
    mapping[i].resize(size);
    for(int j=0; j<size; j++) {
      mapping[i][j] = addresses[j];
    }
  }

  int getAddressSize(){
    return addresses.size();
  }

  long getsizeof(){
    long size = sizeof(RAM);
    size += addresses.size()*sizeof(addr_t);
    size += positions.size()*(sizeof(addr_t)+sizeof(content_t));
    return size;
  }

  // Epoch training: while training the counters live as float in trainPositions;
  // positions (int) is what classification and serialization read.
  void beginTraining(){
    if(trainingMode) return;
    trainPositions.clear();
    for(auto it=positions.begin(); it!=positions.end(); ++it){
      trainPositions[it->first] = (counter_t)it->second;
    }
    ram_t().swap(positions);
    trainingMode = true;
  }

  void endTraining(const double divisor=1){
    if(!trainingMode) return;
    positions.clear();
    for(auto it=trainPositions.begin(); it!=trainPositions.end(); ++it){
      content_t value = (content_t)std::lround(it->second/divisor);
      if(value > 0) positions[it->first] = value;
    }
    ram_train_t().swap(trainPositions);
    trainingMode = false;
  }

  bool isTraining() const{
    return trainingMode;
  }

  template<typename T>
  addr_t getAddress(const T& image) const{
    return getIndex<T>(image);
  }

  bool isPresent(const addr_t addr) const{
    if(ignoreZero && addr == 0) return false;
    if(trainingMode) return trainPositions.find(addr) != trainPositions.end();
    return positions.find(addr) != positions.end();
  }

  counter_t getCounter(const addr_t addr) const{
    if(ignoreZero && addr == 0) return 0;
    if(trainingMode){
      auto it = trainPositions.find(addr);
      return it == trainPositions.end() ? 0 : it->second;
    }
    auto it = positions.find(addr);
    return it == positions.end() ? 0 : (counter_t)it->second;
  }

  bool isActive(const addr_t addr, const double b) const{
    return isPresent(addr) && getCounter(addr) >= b;
  }

  // the vote of the model delivered after dividing the counters by divisor
  bool isActiveDelivered(const addr_t addr, const int b, const double divisor) const{
    return isPresent(addr) && std::lround(getCounter(addr)/divisor) >= b;
  }

  // the value getVote would return after dividing the counters by divisor
  int getVoteDelivered(const addr_t addr, const double divisor) const{
    return isPresent(addr) ? (int)std::lround(getCounter(addr)/divisor) : 0;
  }

  template<typename T>
  void trainF(const T& image){
    trainPositions[getIndex<T>(image)] += 1;
  }

  // m <- m*factor; never negative and the key is erased only when it reaches zero
  void penalize(const addr_t addr, const double factor, const CounterMode mode, EpochRNG& rng){
    if(trainingMode){
      auto it = trainPositions.find(addr);
      if(it == trainPositions.end()) return;
      double value = applyFactor(it->second, factor, mode, rng);
      if(value == 0) trainPositions.erase(it);
      else it->second = (counter_t)value;
    }
    else{
      auto it = positions.find(addr);
      if(it == positions.end()) return;
      double value = applyFactor(it->second, factor, mode, rng);
      content_t rounded = (content_t)std::lround(value);
      if(rounded <= 0) positions.erase(it);
      else it->second = rounded;
    }
  }

  ~RAM(){
    addresses.clear();
    positions.clear();
  }

protected:
  template<typename T>
  addr_t getIndex(const T& image) const{
    addr_t index = 0;
    addr_t p = 1;
    for(unsigned int i=0; i<addresses.size(); i++){
      int bin = image[addresses[i]];
      checkPos(bin);
      index += bin*p;
      p *= base;
    }
    return index;
  }

  template<typename T>
  void train(const T& image){
    addr_t index = getIndex<T>(image);
    auto it = positions.find(index);
    if(it == positions.end()){
      positions.insert(it,std::pair<addr_t,content_t>(index, 1));
    }
    else{
      it->second++;
    }
  }

  template<typename T>
  int getVote(const T& image){
    addr_t index = getIndex<T>(image);
    if(ignoreZero && index == 0)
      return 0;
    auto it = positions.find(index);
    if(it == positions.end()){
      return 0;
    }
    else{
      return it->second;
    }
  }


private:
  std::vector<int> addresses;
  ram_t positions;
  bool ignoreZero;
  int base;
  ram_train_t trainPositions;
  bool trainingMode = false;

  static double applyFactor(const double m, const double factor, const CounterMode mode, EpochRNG& rng){
    double value = m*factor;
    if(mode == COUNTER_INT_STOCHASTIC){
      double low = std::floor(value);
      value = low + (rng.uniform01() < value-low ? 1 : 0);
    }
    return value;
  }

  const std::vector<int> convertToBase(const int number) const{
    std::vector<int> numberConverted(addresses.size());
    int baseNumber = number;
    for(unsigned int i=0; i<numberConverted.size(); i++){
      numberConverted[i] = baseNumber % base;
      baseNumber /= base;
    }
    return numberConverted;
  }

  void checkLimitAddressSize(int addressSize, int basein){
    const addr_t limit = -1;
    if((basein == 2 && addressSize > 64) ||
       (basein != 2 && (addr_t)ipow(basein,addressSize) > limit)){
      throw Exception("The base power to addressSize passed the limit of 2^64!");
    }
  }

  void checkPos(const int code) const{
    if(code >= base){
      throw Exception("The input data has a value bigger than base of addresing!");
    }
  }

  void generateRandomAddresses(int entrySize){
    for(unsigned int i=0; i<addresses.size(); i++){
      addresses[i] = randint(0, entrySize-1);
    }
  }
};
