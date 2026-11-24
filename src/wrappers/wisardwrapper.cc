
class WisardWrapper: public Wisard{
public:
  WisardWrapper(std::string config):Wisard(config){}
  WisardWrapper(int addressSize, py::kwargs kwargs): Wisard(addressSize){

    for(auto arg: kwargs){
      if(std::string(py::str(arg.first)).compare("bleachingActivated") == 0)
        bleachingActivated = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("verbose") == 0)
        verbose = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("ignoreZero") == 0)
        ignoreZero = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("completeAddressing") == 0)
        completeAddressing = arg.second.cast<bool>();

      if (std::string(py::str(arg.first)).compare("mapping") == 0)
        mapping = arg.second.cast<std::map<std::string, std::vector<std::vector<int>>>>();

      if(std::string(py::str(arg.first)).compare("indexes") == 0)
        indexes = arg.second.cast<std::vector<int>>();

      if(std::string(py::str(arg.first)).compare("base") == 0)
        base = arg.second.cast<int>();

      if(std::string(py::str(arg.first)).compare("searchBestConfidence") == 0)
        searchBestConfidence = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("returnConfidence") == 0)
        returnConfidence = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("returnActivationDegree") == 0)
        returnActivationDegree = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("returnClassesDegrees") == 0)
        returnClassesDegrees = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("confidence") == 0)
        confidence = arg.second.cast<int>();

      if(std::string(py::str(arg.first)).compare("returnBleachingInfo") == 0)
        returnBleachingInfo = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("bleaching") == 0)
        bleachingMode = arg.second.cast<std::string>();

      if(std::string(py::str(arg.first)).compare("confidenceTolerance") == 0)
        confidenceTolerance = arg.second.cast<double>();

      if(std::string(py::str(arg.first)).compare("freqTolerance") == 0)
        freqTolerance = arg.second.cast<double>();

      if(std::string(py::str(arg.first)).compare("dabRemoval") == 0)
        dabRemoval = arg.second.cast<std::string>();

      if(std::string(py::str(arg.first)).compare("multiBleaching") == 0)
        multiBleaching = arg.second.cast<bool>();

      if(std::string(py::str(arg.first)).compare("confidenceThreshold") == 0)
        confidenceThreshold = arg.second.cast<double>();

      if(std::string(py::str(arg.first)).compare("tieTolerance") == 0)
        tieTolerance = arg.second.cast<double>();

      if(std::string(py::str(arg.first)).compare("nextCut") == 0)
        nextCut = arg.second.cast<std::string>();

      if(std::string(py::str(arg.first)).compare("maxRounds") == 0)
        maxRounds = arg.second.cast<int>();
    }
    checkBleachingOptions();
  }

  // changes the bleaching of classify on a trained model (the counters are not touched), so
  // several policies can be compared on the same network. Unknown options are an error and a
  // rejected value leaves the model as it was.
  void pySetBleaching(py::kwargs kwargs){
    const bool oldActivated = bleachingActivated, oldMulti = multiBleaching;
    const int oldConfidence = confidence, oldRounds = maxRounds;
    const std::string oldMode = bleachingMode, oldRemoval = dabRemoval, oldCut = nextCut;
    const double oldTolerance = confidenceTolerance, oldFreq = freqTolerance, oldThreshold = confidenceThreshold, oldTie = tieTolerance;
    try{
      for(auto arg: kwargs){
        const std::string key = py::str(arg.first);
        if(key == "bleaching") bleachingMode = arg.second.cast<std::string>();
        else if(key == "bleachingActivated") bleachingActivated = arg.second.cast<bool>();
        else if(key == "confidence") confidence = arg.second.cast<int>();
        else if(key == "confidenceTolerance") confidenceTolerance = arg.second.cast<double>();
        else if(key == "freqTolerance") freqTolerance = arg.second.cast<double>();
        else if(key == "dabRemoval") dabRemoval = arg.second.cast<std::string>();
        else if(key == "multiBleaching") multiBleaching = arg.second.cast<bool>();
        else if(key == "confidenceThreshold") confidenceThreshold = arg.second.cast<double>();
        else if(key == "tieTolerance") tieTolerance = arg.second.cast<double>();
        else if(key == "nextCut") nextCut = arg.second.cast<std::string>();
        else if(key == "maxRounds") maxRounds = arg.second.cast<int>();
        else throw Exception("Unknown option for setBleaching!");
      }
      checkBleachingOptions();
    }
    catch(...){
      bleachingActivated = oldActivated; multiBleaching = oldMulti; confidence = oldConfidence; maxRounds = oldRounds;
      bleachingMode = oldMode; dabRemoval = oldRemoval; nextCut = oldCut;
      confidenceTolerance = oldTolerance; freqTolerance = oldFreq; confidenceThreshold = oldThreshold; tieTolerance = oldTie;
      throw;
    }
  }

  // the confidence is the one of the policy (for iterative, Bleaching::getConfidence as before)
  void setClassifyOutput(py::list& labels, int i, float numberOfRAMS, BleachingResult& result){
    const std::string& aClass = result.label;
    const bool asDict = returnConfidence || returnActivationDegree || returnClassesDegrees || returnBleachingInfo;
    if(asDict){
      labels[i] = py::dict(py::arg("class")=aClass);
    }

    if(returnActivationDegree){
      float activationDegree = result.candidates[aClass]/numberOfRAMS;
      labels[i]["activationDegree"]=activationDegree;
    }

    if(returnConfidence){
      float confidence = (float)result.confidence;
      labels[i]["confidence"]=confidence;
    }

    if(returnClassesDegrees){
      labels[i]["classesDegrees"] = getClassesDegrees(result.candidates);
    }

    if(returnBleachingInfo){
      labels[i]["steps"] = result.steps;
      labels[i]["finalThreshold"] = result.finalThreshold;
      if(!result.thresholds.empty()){  // bbleaching and multiBleaching: every round and its confidence
        labels[i]["thresholds"] = result.thresholds;
        labels[i]["confidences"] = result.confidences;
      }
    }

    if(!asDict){
      labels[i] = aClass;
    }
  }

  py::list pyClassify(const std::vector<std::vector<int>>& images){
    return _pyClassify<std::vector<std::vector<int>>>(images);
  }

  py::list pyClassify(const DataSet& images){
    return _pyClassify<DataSet>(images);
  }

  py::dict pyTrainEpochs(const std::vector<std::vector<int>>& images, const std::vector<std::string>& labels,
                         const std::vector<std::vector<int>>& valImages, const std::vector<std::string>& valLabels,
                         py::kwargs kwargs){
    return reportToDict(trainEpochs(images, labels, valImages, valLabels, EpochConfig(kwargsToJSON(kwargs))));
  }

  py::dict pyTrainEpochs(const DataSet& train, const DataSet& val, py::kwargs kwargs){
    return reportToDict(trainEpochs(train, val, EpochConfig(kwargsToJSON(kwargs))));
  }

  void pyPenalize(const std::string& label, const std::vector<std::pair<int,addr_t>>& addrs, const double factor, py::kwargs kwargs){
    CounterMode mode = COUNTER_FLOAT;
    unsigned int seed = 0;
    for(auto arg: kwargs){
      if(std::string(py::str(arg.first)).compare("counterMode") == 0)
        mode = EpochConfig::parseCounterMode(arg.second.cast<std::string>());
      else if(std::string(py::str(arg.first)).compare("seed") == 0)
        seed = arg.second.cast<unsigned int>();
      else
        throw Exception("Unknown option for penalize!");
    }
    penalize(label, addrs, factor, mode, seed);
  }

protected:
  static py::dict reportToDict(const EpochReport& report){
    py::list perEpoch;
    for(unsigned int i=0; i<report.perEpoch.size(); i++){
      perEpoch.append(py::cast(report.perEpoch[i]));
    }
    py::dict out;
    out["bestEpoch"] = report.bestEpoch;
    out["perEpoch"] = perEpoch;
    return out;
  }

  static nl::json kwargsToJSON(py::kwargs& kwargs){
    nl::json c = nl::json::object();
    for(auto arg: kwargs){
      std::string key = py::str(arg.first);
      if(py::isinstance<py::bool_>(arg.second)) c[key] = arg.second.cast<bool>();
      else if(py::isinstance<py::int_>(arg.second)) c[key] = arg.second.cast<long long>();
      else if(py::isinstance<py::float_>(arg.second)) c[key] = arg.second.cast<double>();
      else if(py::isinstance<py::str>(arg.second)) c[key] = arg.second.cast<std::string>();
      else throw Exception("The epoch training options must be bool, int, float or str!");
    }
    return c;
  }

  py::list getClassesDegrees(std::map<std::string, int> candidates) const{
    float total = 0;
    int index = 0;
    py::list output(candidates.size());
    for(std::map<std::string, int>::iterator i=candidates.begin(); i!=candidates.end(); ++i) total += i->second;
    if(total == 0) total=1;
    for(std::map<std::string, int>::iterator i=candidates.begin(); i!=candidates.end(); ++i){
      output[index] = py::dict(py::arg("class")=i->first, py::arg("degree")=(i->second/total));
      index++;
    }
    return output;
  }

  template<typename T>
  py::list _pyClassify(const T& images){
    float numberOfRAMS = calculateNumberOfRams(images[0].size(), addressSize, completeAddressing);

    py::list labels(images.size());
    std::unique_ptr<BleachingPolicy> policy = makeBleachingPolicy();
    const std::map<std::string,int> sizes = classSizes();
    for(unsigned int i=0; i<images.size(); i++){
      if(verbose) std::cout << "\rclassifying " << i+1 << " of " << images.size();
      BleachingResult result = decide(images[i], *policy, sizes);
      setClassifyOutput(labels, i, numberOfRAMS, result);
    }
    if(verbose) std::cout << "\r" << std::endl;
    return labels;
  }
};
