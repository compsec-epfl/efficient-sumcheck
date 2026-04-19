window.BENCHMARK_DATA = {
  "lastUpdate": 1776615784841,
  "repoUrl": "https://github.com/compsec-epfl/efficient-sumcheck",
  "entries": {
    "Sumcheck Benchmarks (scalar)": [
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
          "id": "5f79fc443969ec7447824dde91f42508ec66f98f",
          "message": "Major rewrite (#99)\n\n* canonical rewrite: SumcheckProver trait, generic field, SIMD, streaming\n\n* fmt\n\n* fix ci\n\n* readme\n\n* no std, readme + docs\n\n* fmt\n\n* readme\n\n* force oracle check\n\n* revert oracle check thing\n\n* eq evals opts\n\n* changelog\n\n* p3 compatibility example",
          "timestamp": "2026-04-19T18:18:51+02:00",
          "tree_id": "45812faf8acb6a5ac96b810daa8adcb8fa9a1f6e",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/5f79fc443969ec7447824dde91f42508ec66f98f"
        },
        "date": 1776615784531,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 269744,
            "range": "± 6028",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 2862828,
            "range": "± 8756",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 41769085,
            "range": "± 133292",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1338981,
            "range": "± 6120",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 17155661,
            "range": "± 32214",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 268080924,
            "range": "± 127456",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 1119853,
            "range": "± 6812",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 15658187,
            "range": "± 37637",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 279965330,
            "range": "± 96719",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 9260979,
            "range": "± 45344",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 134062294,
            "range": "± 130390",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 2158174815,
            "range": "± 8863519",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 84736,
            "range": "± 486",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 1115402,
            "range": "± 10391",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 14924232,
            "range": "± 239529",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 503797,
            "range": "± 1451",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 7726451,
            "range": "± 18239",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 120213155,
            "range": "± 172404",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}