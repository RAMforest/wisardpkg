
// Bleaching policies of Wisard::classify (dab-dev.md, part 1). A policy receives the counters
// read in every RAM of every discriminator for one input and decides the class:
//   iterative  the bleaching of the package (Bleaching): b = 1, 2, 3, ... while there is a tie
//   dab        jumps straight to the threshold that changes the votes of the tied classes, until
//              the confidence c = (r1 - r2)/r1 reaches confidenceTolerance
//   rdab       dab with a second criterion, tested before each new threshold: the sum of the
//              voting counters over the class size
// A counter votes when it is >= the threshold; zeros never vote.

struct BleachingResult{
  std::string label;
  std::map<std::string,int> candidates;  // votes of every class at the deciding step
  double confidence = 0;
  int steps = 0;           // how many times the votes were counted
  int finalThreshold = 1;  // smallest counter that votes at the deciding step
};

class BleachingPolicy{
public:
  virtual ~BleachingPolicy(){}
  // classSize: training examples of every class (Discriminator count); only rdab reads it
  virtual BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const = 0;
};

class IterativeBleaching: public BleachingPolicy{
public:
  IterativeBleaching(const bool activated, const int confidence): activated(activated), confidence(confidence){}

  BleachingResult resolve(std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize) const{
    BleachingResult r;
    r.candidates = Bleaching::makeConfidenceless(votes, activated, confidence, r.steps, r.finalThreshold);
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

protected:
  // freqTolerance < 0: dab; otherwise rdab
  BleachingResult run(const std::map<std::string,std::vector<int>>& votes, const std::map<std::string,int>& classSize, const double freqTolerance) const{
    BleachingResult previous;
    int keepFrom = 1;
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

private:
  double freqTolerance;
};
