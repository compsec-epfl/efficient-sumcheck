window.BENCHMARK_DATA = {
  "lastUpdate": 1776630458460,
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
      },
      {
        "commit": {
          "author": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "committer": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "distinct": true,
          "id": "9153e8f7d29788159af9791d8262f20b8864fd15",
          "message": "update readme",
          "timestamp": "2026-04-19T18:25:34+02:00",
          "tree_id": "30faf457b617c83724bd3b781c211d00c525a8c5",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/9153e8f7d29788159af9791d8262f20b8864fd15"
        },
        "date": 1776616159534,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 242109,
            "range": "± 633",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 2625796,
            "range": "± 13633",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 36352669,
            "range": "± 66243",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1278047,
            "range": "± 7034",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 15766364,
            "range": "± 27293",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 246147594,
            "range": "± 1563819",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 1019302,
            "range": "± 5977",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 14082608,
            "range": "± 74552",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 235037730,
            "range": "± 3145215",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 8385207,
            "range": "± 17504",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 123669998,
            "range": "± 123996",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 1990314999,
            "range": "± 794456",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 78492,
            "range": "± 385",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 1049774,
            "range": "± 6989",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 12714479,
            "range": "± 132305",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 473365,
            "range": "± 2810",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 7078123,
            "range": "± 9804",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 110064872,
            "range": "± 82690",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "committer": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "distinct": true,
          "id": "a9f73cf1eb4337a0f9cecbdbb1990fe3e59d2ab9",
          "message": "update slides",
          "timestamp": "2026-04-19T18:41:40+02:00",
          "tree_id": "2cb0281aa0c6dc44ae9e20aa195551c3d36f3575",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/a9f73cf1eb4337a0f9cecbdbb1990fe3e59d2ab9"
        },
        "date": 1776617132803,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 244877,
            "range": "± 5798",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 2619208,
            "range": "± 14260",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 36496447,
            "range": "± 334124",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1283947,
            "range": "± 10640",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 15869728,
            "range": "± 284948",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 246113480,
            "range": "± 721014",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 1038271,
            "range": "± 7636",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 14189974,
            "range": "± 40237",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 235017953,
            "range": "± 375023",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 8388477,
            "range": "± 24317",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 123435551,
            "range": "± 496190",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 1989643667,
            "range": "± 765228",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 80231,
            "range": "± 614",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 1039576,
            "range": "± 10899",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 12717025,
            "range": "± 173617",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 475339,
            "range": "± 3114",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 7222123,
            "range": "± 27930",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 110034048,
            "range": "± 2558267",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "committer": {
            "email": "1497456+z-tech@users.noreply.github.com",
            "name": "Andrew Z",
            "username": "z-tech"
          },
          "distinct": true,
          "id": "36976a74fa08ba8ba3a6cfbd6271145bbabcc889",
          "message": "update security",
          "timestamp": "2026-04-19T22:23:40+02:00",
          "tree_id": "79adaa7a49fa24b16285db4c385dedccc7694c97",
          "url": "https://github.com/compsec-epfl/efficient-sumcheck/commit/36976a74fa08ba8ba3a6cfbd6271145bbabcc889"
        },
        "date": 1776630457727,
        "tool": "cargo",
        "benches": [
          {
            "name": "multilinear/F64/2^16",
            "value": 245541,
            "range": "± 4646",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^20",
            "value": 2626976,
            "range": "± 10098",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64/2^24",
            "value": 36445325,
            "range": "± 107247",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^16",
            "value": 1279861,
            "range": "± 9524",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^20",
            "value": 15867916,
            "range": "± 105194",
            "unit": "ns/iter"
          },
          {
            "name": "multilinear/F64Ext3/2^24",
            "value": 245979494,
            "range": "± 1035805",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^16",
            "value": 1030331,
            "range": "± 7449",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^20",
            "value": 14186607,
            "range": "± 87198",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64/2^24",
            "value": 234975740,
            "range": "± 276290",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^16",
            "value": 8391658,
            "range": "± 36116",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^20",
            "value": 123504325,
            "range": "± 120090",
            "unit": "ns/iter"
          },
          {
            "name": "inner_product/F64Ext3/2^24",
            "value": 1989702735,
            "range": "± 1317867",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^16",
            "value": 79690,
            "range": "± 628",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^20",
            "value": 1046562,
            "range": "± 8656",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64/2^24",
            "value": 12718004,
            "range": "± 30205",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^16",
            "value": 472873,
            "range": "± 8451",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^20",
            "value": 7188709,
            "range": "± 17525",
            "unit": "ns/iter"
          },
          {
            "name": "fold/F64Ext3/2^24",
            "value": 110037781,
            "range": "± 48147",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}