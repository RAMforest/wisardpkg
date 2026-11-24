
// Bleaching policies of Wisard::classify (dab-dev.md, part 1). A policy receives the counters
// read in every RAM of every discriminator for one input and decides the class:
//   iterative  the bleaching of the package (Bleaching): b = 1, 2, 3, ... while there is a tie
//   dab        jumps straight to the threshold that changes the votes of the tied classes, until
//              the confidence c = (r1 - r2)/r1 reaches confidenceTolerance
//   rdab       dab with a second criterion, tested before each new threshold: the sum of the
//              voting counters over the class size
//   bbleaching binary search of one threshold in [1, B_k) with the geometric mean (Carvalho et
//              al., ESANN 2013); B_k is the largest counter read
// Multi-bleaching (MultiBleaching) is a layer over iterative, dab or bbleaching: the base decides
// at a threshold; while the confidence of that decision is not above confidenceThreshold, the
// base runs again from the next threshold; the score of a class is the sum of its votes at the
// deciding thresholds of all the rounds.
// A counter votes when it is >= the threshold; zeros never vote.

struct BleachingResult{
  std::string label;
  std::map<std::string,int> candidates;  // votes of every class at the deciding step
  double confidence = 0;
  int steps = 0;           // how many times the votes were counted
  int finalThreshold = 1;  // smallest counter that votes at the deciding step
  // bbleaching and multi-bleaching: the deciding threshold of every round and the confidence
  // (r1 - r2)/r1 of the votes at it, 0 on ties; one round without multi-bleaching
  std::vector<int> thresholds;
  std::vector<double> confidences;
};

class BleachingPolicy{
public:
  virtual ~BleachingPolicy(){}
  // classSize: training examples of every class (Discriminator count); only rdab reads it
  virtual BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const = 0;
  // the same decision with the search starting at the threshold start, as if the counters below
  // it were not there (a round of multi-bleaching); resolve is resolveFrom(..., 1)
  virtual BleachingResult resolveFrom(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const int start) const{
    throw Exception("This bleaching can not be used with multiBleaching!");
  }
};

class IterativeBleaching: public BleachingPolicy{
public:
  IterativeBleaching(const bool activated, const int confidence): activated(activated), confidence(confidence){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    return resolveFrom(votes, classSize, 1);
  }

  BleachingResult resolveFrom(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const int start) const{
    BleachingResult r;
    r.candidates = Bleaching::makeConfidenceless(votes, activated, confidence, r.steps, r.finalThreshold, start);
    r.label = Bleaching::getBiggestCandidate(r.candidates);
    auto it = r.candidates.find(r.label);
    r.confidence = Bleaching::getConfidence(r.candidates, it == r.candidates.end() ? 0 : it->second);
    return r;
  }

private:
  bool activated;
  int confidence;
};

class DAB: public BleachingPolicy{
public:
  // inclusive: the next threshold drops the counters <= tau_b (otherwise only the ones < tau_b)
  DAB(const double confidenceTolerance, const bool inclusive): confidenceTolerance(confidenceTolerance), inclusive(inclusive){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    return run(votes, classSize, -1);
  }

  BleachingResult resolveFrom(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const int start) const{
    return run(votes, classSize, -1, start);
  }

protected:
  // freqTolerance < 0: dab; otherwise rdab. start: the first threshold (1 for classify)
  BleachingResult run(const std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const double freqTolerance, const int start=1) const{
    BleachingResult previous;
    int keepFrom = start;
    for(int step=1; ; step++){
      BleachingResult r;
      r.steps = step;
      r.finalThreshold = keepFrom;
      double r1 = 0, r2 = 0;
      for(auto d=votes.begin(); d!=votes.end(); ++d){
        int n = 0;
        for(unsigned int j=0; j<d->second.size(); j++){
          if(d->second[j] >= keepFrom) n++;
        }
        r.candidates[d->first] = n;
        top2(d->first, n, r.label, r1, r2);
      }

      if(r1 == 0){  // no counter votes: at the first step there is nothing to decide with
        if(step == 1) return r;
        previous.steps = step;  // the last threshold that still had votes decides
        return previous;
      }
      r.confidence = (r1-r2)/r1;
      if(r.confidence >= confidenceTolerance) return r;

      if(freqTolerance >= 0){
        std::string fLabel;
        double f1 = 0, f2 = 0;
        for(auto d=votes.begin(); d!=votes.end(); ++d){
          double sum = 0;
          for(unsigned int j=0; j<d->second.size(); j++){
            if(d->second[j] >= keepFrom) sum += d->second[j];
          }
          auto size = classSize.find(d->first);
          top2(d->first, size != classSize.end() && size->second > 0 ? sum/size->second : 0, fLabel, f1, f2);
        }
        const double cf = f1 > 0 ? (f1-f2)/f1 : 0;
        if(cf >= freqTolerance){
          r.label = fLabel;
          r.confidence = cf;
          return r;
        }
      }

      // tau_b: the largest of the smallest voting counter of the tied classes (the ones within
      // confidenceTolerance of the leader); the same threshold is applied to every class
      int tauB = 0;
      for(auto d=votes.begin(); d!=votes.end(); ++d){
        const int n = r.candidates[d->first];
        if(n == 0 || (r1-n)/r1 >= confidenceTolerance) continue;
        int smallest = -1;
        for(unsigned int j=0; j<d->second.size(); j++){
          const int v = d->second[j];
          if(v >= keepFrom && (smallest < 0 || v < smallest)) smallest = v;
        }
        tauB = std::max(tauB, smallest);
      }
      const int next = inclusive ? tauB+1 : tauB;
      if(next <= keepFrom) return r;  // the threshold would not grow: the tie stays

      previous = r;
      keepFrom = next;
    }
  }

  // ties go to the last label, as in Bleaching::getBiggestCandidate
  static void top2(const std::string& label, const double value, std::string& first, double& v1, double& v2){
    if(value >= v1){
      v2 = v1;
      v1 = value;
      first = label;
    }
    else if(value > v2){
      v2 = value;
    }
  }

  double confidenceTolerance;
  bool inclusive;
};

class RDAB: public DAB{
public:
  RDAB(const double confidenceTolerance, const double freqTolerance, const bool inclusive): DAB(confidenceTolerance, inclusive), freqTolerance(freqTolerance){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    return run(votes, classSize, freqTolerance);
  }

  // the second criterion decides by the counters themselves, so a round from a later threshold is not defined
  BleachingResult resolveFrom(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const int start) const{
    throw Exception("rdab can not be used with multiBleaching!");
  }

private:
  double freqTolerance;
};

// the two largest values; ties go to the last label, as in Bleaching::getBiggestCandidate
static inline void bleachingTop2(const std::string& label, const double value, std::string& first, double& v1, double& v2){
  if(value >= v1){
    v2 = v1;
    v1 = value;
    first = label;
  }
  else if(value > v2){
    v2 = value;
  }
}

static inline int largestCounter(const std::map<std::string,std::vector<int>>& votes){
  int biggest = 0;
  for(auto d=votes.begin(); d!=votes.end(); ++d){
    for(unsigned int j=0; j<d->second.size(); j++){
      if(d->second[j] > biggest) biggest = d->second[j];
    }
  }
  return biggest;
}

// Binary search of the bleaching threshold (B-bleaching). On [lo, hi] the reference is the
// largest response at lo; the probe is floor(sqrt(lo*hi)): with a tie (confidence <=
// tieTolerance) the search goes up, without a tie but with a smaller largest response it goes
// down, otherwise the probe is accepted. When the interval ends, the probe with the best
// confidence decides. hi is B_k - 1, B_k the largest counter read.
class BBleachingPolicy: public BleachingPolicy{
public:
  // referenceCut: a search that starts after the first threshold (a later round of
  // multi-bleaching) accepts a probe only if its largest response is the one at the start of the
  // interval ("ref"); otherwise the first probe without a tie ("untie")
  BBleachingPolicy(const double tieTolerance, const bool referenceCut=true): tieTolerance(tieTolerance), referenceCut(referenceCut){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    return resolveFrom(votes, classSize, 1);
  }

  // start 1: the threshold 1 is accepted when it has no tie (no bleaching when it is not
  // needed); later starts are only the reference of the search
  BleachingResult resolveFrom(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const int start) const{
    BleachingResult r;
    r.steps = 0;
    const int hi = std::max(std::max(largestCounter(votes)-1, 1), start);
    const Probe p = search(votes, start, hi, start == 1, start == 1 || referenceCut, r.steps);
    r.label = p.label;
    r.candidates = p.candidates;
    r.confidence = p.confidence;
    r.finalThreshold = p.threshold;
    r.thresholds.push_back(p.threshold);
    r.confidences.push_back(p.confidence);
    return r;
  }

protected:
  struct Probe{
    int threshold = 1;
    std::map<std::string,int> candidates;
    std::string label;
    double first = 0, second = 0, confidence = 0;
  };

  static Probe count(const std::map<std::string,std::vector<int>>& votes, const int threshold){
    Probe p;
    p.threshold = threshold;
    for(auto d=votes.begin(); d!=votes.end(); ++d){
      int n = 0;
      for(unsigned int j=0; j<d->second.size(); j++){
        if(d->second[j] >= threshold) n++;
      }
      p.candidates[d->first] = n;
      bleachingTop2(d->first, n, p.label, p.first, p.second);
    }
    p.confidence = p.first > 0 ? (p.first-p.second)/p.first : 0;
    return p;
  }

  Probe search(const std::map<std::string,std::vector<int>>& votes, int lo, int hi, const bool acceptLo, const bool requireReference, int& steps) const{
    Probe best = count(votes, lo);
    steps++;
    const double reference = best.first;
    if(acceptLo){
      if(best.confidence > tieTolerance) return best;
      lo++;
    }
    while(lo <= hi){
      const int mid = std::min(hi, std::max(lo, (int)std::floor(std::sqrt((double)lo*(double)hi))));
      const Probe p = count(votes, mid);
      steps++;
      const bool tied = p.confidence <= tieTolerance;
      if(!tied && (!requireReference || p.first == reference)) return p;
      if(p.confidence > best.confidence) best = p;
      if(tied) lo = mid+1;
      else hi = mid-1;
    }
    return best;
  }

  double tieTolerance;
  bool referenceCut;
};

// Multi-bleaching over a base policy. Round 1 is the base as it is. While the confidence of the
// round, (r1 - r2)/r1 of the votes at its deciding threshold (0 on ties), is not above
// confidenceThreshold and some counter is above that threshold, the base runs again from the
// next threshold. The score of a class is the sum of its votes at the deciding thresholds, so a
// RAM with a large counter scores once per round it survives. candidates and confidence of the
// result are the ones of the summed scores; thresholds and confidences, the ones of the rounds;
// steps, the vote counts of all the rounds.
class MultiBleaching: public BleachingPolicy{
public:
  MultiBleaching(std::unique_ptr<BleachingPolicy> base, const double confidenceThreshold, const int maxRounds):
    base(std::move(base)), confidenceThreshold(confidenceThreshold), maxRounds(maxRounds){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    BleachingResult r;
    r.steps = 0;
    for(auto d=votes.begin(); d!=votes.end(); ++d) r.candidates[d->first] = 0;
    const int last = largestCounter(votes);
    int start = 1;
    for(int round=0; round<maxRounds && (round == 0 || start <= last); round++){
      const BleachingResult p = base->resolveFrom(votes, classSize, start);
      std::string leader;
      double r1 = 0, r2 = 0;
      for(auto c=p.candidates.begin(); c!=p.candidates.end(); ++c){
        r.candidates[c->first] += c->second;
        bleachingTop2(c->first, c->second, leader, r1, r2);
      }
      const double confidence = r1 > 0 ? (r1-r2)/r1 : 0;
      r.steps += p.steps;
      r.finalThreshold = p.finalThreshold;
      r.thresholds.push_back(p.finalThreshold);
      r.confidences.push_back(confidence);
      if(confidence > confidenceThreshold) break;
      start = p.finalThreshold+1;
    }
    double s1 = 0, s2 = 0;
    for(auto c=r.candidates.begin(); c!=r.candidates.end(); ++c) bleachingTop2(c->first, c->second, r.label, s1, s2);
    r.confidence = s1 > 0 ? (s1-s2)/s1 : 0;
    return r;
  }

private:
  std::unique_ptr<BleachingPolicy> base;
  double confidenceThreshold;
  int maxRounds;
};
