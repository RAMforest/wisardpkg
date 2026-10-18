typedef unsigned long long addr_t;
typedef unsigned long long index_size_t;
typedef char bin_t;
typedef int content_t;
typedef std::unordered_map<addr_t, content_t> ram_t;
typedef float counter_t;
typedef std::unordered_map<addr_t, counter_t> ram_train_t;

const std::string ramdata_sufix = ".wdpkg";
const std::string dataset_sufix = ".wpkds";
