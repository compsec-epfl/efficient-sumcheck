window.BENCHMARK_DATA = {
  "lastUpdate": 1777384941933,
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
      },
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
          "id": "fe5c8dac7e052776f8c99afcdcef83766240016f",
          "message": "allow to owned call on coeff prover (#102)",
          "timestamp": "2026-04-28T15:58:09+02:00",
          "tree_id": "5bcdbd30e0bc8bb413630161c67fd51a5a123be7",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/fe5c8dac7e052776f8c99afcdcef83766240016f"
        },
        "date": 1777384941260,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 306802,
            "range": "± 6062",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 3532787,
            "range": "± 23458",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 53096484,
            "range": "± 165718",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1346029,
            "range": "± 20176",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 17223633,
            "range": "± 68140",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 269722586,
            "range": "± 1279647",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 856856,
            "range": "± 7571",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 10933742,
            "range": "± 54001",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 159792907,
            "range": "± 549204",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 7266746,
            "range": "± 43682",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 106675971,
            "range": "± 346268",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 1676301361,
            "range": "± 3475704",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 37966,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 709165,
            "range": "± 5762",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 12567598,
            "range": "± 165018",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 573171,
            "range": "± 3348",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 7767655,
            "range": "± 39680",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 120865926,
            "range": "± 125450",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}