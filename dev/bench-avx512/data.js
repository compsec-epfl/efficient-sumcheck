window.BENCHMARK_DATA = {
  "lastUpdate": 1776882739892,
  "repoUrl": "https://github.com/compsec-epfl/efficient-sumcheck",
  "entries": {
    "Sumcheck Benchmarks (avx512)": [
      {
        "commit": {
          "author": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Zitek-Estrada",
            "username": "z-tech"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4273d11a3dd531ffcfd64d6f094b10ba15160fcd",
          "message": "chkpt (#100)\n\n* chkpt\n\n* update docs\n\n* fix ci",
          "timestamp": "2026-04-22T20:28:17+02:00",
          "tree_id": "a463f708f52b0485ec154aef2eeb2319ccb8fe7d",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/4273d11a3dd531ffcfd64d6f094b10ba15160fcd"
        },
        "date": 1776882738992,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 260760,
            "range": "± 7461",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 2813089,
            "range": "± 11236",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 40571940,
            "range": "± 206080",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1105085,
            "range": "± 41308",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 13591045,
            "range": "± 65686",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 210811937,
            "range": "± 3390823",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 624927,
            "range": "± 6366",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 7963514,
            "range": "± 33911",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 140476102,
            "range": "± 162259",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 6455684,
            "range": "± 54595",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 95388382,
            "range": "± 1054448",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 1508517840,
            "range": "± 14212429",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 35931,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 724951,
            "range": "± 2033",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 9788320,
            "range": "± 23063",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 395410,
            "range": "± 1018",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 6127954,
            "range": "± 14004",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 93924694,
            "range": "± 192690",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}