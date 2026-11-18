
// fn(begin, end, worker) over [0, n) in blocks of chunk items. Every worker (0 .. threads-1)
// runs in its own thread and takes the next free block when it ends one, so slower cores
// (e.g. the efficiency cores of a hybrid CPU) take fewer blocks; fn may be called many times
// with the same worker, never at the same time. With one thread, or one block, it all runs
// here. An exception of any worker is thrown again after all of them end.
template<typename F>
void parallelFor(const size_t n, const int threads, F fn, const size_t chunk=256){
  const size_t blocks = (n + chunk - 1)/chunk;
  const size_t workers = std::min((size_t)std::max(threads, 1), blocks);
  if(workers <= 1){
    if(n > 0) fn((size_t)0, n, (size_t)0);
    return;
  }
  std::atomic<size_t> next(0);
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(workers);
  for(size_t w=0; w<workers; w++){
    pool.emplace_back([&fn, &errors, &next, w, n, chunk](){
      try{
        for(size_t begin = next.fetch_add(chunk); begin < n; begin = next.fetch_add(chunk)){
          fn(begin, std::min(n, begin + chunk), w);
        }
      }
      catch(...){
        errors[w] = std::current_exception();
      }
    });
  }
  for(size_t w=0; w<workers; w++) pool[w].join();
  for(size_t w=0; w<workers; w++){
    if(errors[w]) std::rethrow_exception(errors[w]);
  }
}

// threads of the epoch training: 0 means every core of the machine
inline int epochThreads(const int threads){
  if(threads > 0) return threads;
  const unsigned int cores = std::thread::hardware_concurrency();
  return cores > 0 ? (int)cores : 1;
}

// The address of every example of a set on every RAM of a discriminator, computed once for
// all the epochs of trainEpochs (an address only depends on the input and on the mapping).
// Every address takes the fewest bytes that hold the largest address of the RAMs (1 byte for
// tuples of up to 8 bits). A table larger than the memory left in the budget is not built:
// the addresses are then computed at each use.
class AddressTable{
public:
  template<typename S>
  static std::shared_ptr<AddressTable> build(const S& samples, const Discriminator& d, const int threads, size_t& budget){
    const size_t rams = d.getNumberOfRAMS();
    const int width = addressWidth(d);
    const size_t bytes = samples.size()*rams*width;
    if(bytes > budget) return std::shared_ptr<AddressTable>();
    budget -= bytes;

    std::shared_ptr<AddressTable> table(new AddressTable(rams, width));
    table->data.resize(bytes);
    AddressTable* t = table.get();
    parallelFor(samples.size(), threads, [&samples, &d, t](size_t begin, size_t end, size_t){
      std::vector<addr_t> row(t->rams);
      for(size_t i=begin; i<end; i++){
        d.fillAddresses(samples.image(i), row.data());
        t->store(i, row.data());
      }
    });
    return table;
  }

  // out[r]: the address of example i on RAM r
  void row(const size_t i, addr_t* out) const{
    const unsigned char* p = data.data() + i*rams*width;
    switch(width){
      case 1:
        for(size_t r=0; r<rams; r++) out[r] = p[r];
        break;
      case 2:
        for(size_t r=0; r<rams; r++){ uint16_t v; std::memcpy(&v, p+2*r, 2); out[r] = v; }
        break;
      case 4:
        for(size_t r=0; r<rams; r++){ uint32_t v; std::memcpy(&v, p+4*r, 4); out[r] = v; }
        break;
      default:
        for(size_t r=0; r<rams; r++){ uint64_t v; std::memcpy(&v, p+8*r, 8); out[r] = (addr_t)v; }
    }
  }

private:
  size_t rams;
  int width;
  std::vector<unsigned char> data;

  AddressTable(const size_t rams, const int width): rams(rams), width(width){}

  void store(const size_t i, const addr_t* row){
    unsigned char* p = data.data() + i*rams*width;
    for(size_t r=0; r<rams; r++){
      const uint64_t v = (uint64_t)row[r];
      switch(width){
        case 1: p[r] = (unsigned char)v; break;
        case 2: { uint16_t u = (uint16_t)v; std::memcpy(p+2*r, &u, 2); break; }
        case 4: { uint32_t u = (uint32_t)v; std::memcpy(p+4*r, &u, 4); break; }
        default: std::memcpy(p+8*r, &v, 8);
      }
    }
  }

  // bytes of the largest address, base^(tuple size) - 1, of the RAMs of d
  static int addressWidth(const Discriminator& d){
    int width = 1;
    for(int r=0; r<d.getNumberOfRAMS(); r++){
      const RAM& ram = d.getRAM(r);
      const unsigned long long base = ram.getBase();
      unsigned long long capacity = 1;
      bool overflow = false;
      for(size_t k=0; k<ram.getAddresses().size(); k++){
        if(capacity > ULLONG_MAX/base){ overflow = true; break; }
        capacity *= base;
      }
      const unsigned long long largest = overflow ? ULLONG_MAX : capacity-1;
      while(width < 8 && (width == 4 ? largest > 0xFFFFFFFFULL : largest >= (1ULL << (8*width)))){
        width *= 2;
      }
    }
    return width;
  }
};
