export const meta = {
  name: 'persubject-graphrag',
  description: 'GraphRAG 근거(보기 처방 구성·주치·계통) + 과목별 변증추론 + 자기일관성으로 국시 풀이',
  phases: [{ title: 'GraphRAG', detail: '지식그래프 근거로 과목별 추론(자기일관성)' }],
}
// args: { start, end, k }  — eval_input.jsonl(idx·과목·question·options·evidence) 사용
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
let IDXS = Array.isArray(A.idxs) ? A.idxs : []
if (!IDXS.length && A.start != null && A.end != null) { for (let i = A.start; i <= A.end; i++) IDXS.push(i) }
const K = A.k || 1
const EVAL = A.evalfile || 'D:\\tmp\\persubj\\eval_input.jsonl'  // 해설RAG는 eval_input_hae.jsonl 지정

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['idx', 'pred'],
  properties: { idx: { type: 'integer' }, pred: { type: 'integer', minimum: 1, maximum: 5 } },
}

const prompt = (idx, s) => `너는 한의사·한약사 국가시험 전 과목에 정통한 한의학 전문가다. 5지선다 한 문항을 최대한 정확히 푼다 (샘플 ${s + 1}).

파일 \`${EVAL}\` 에서 idx==${idx} 인 줄을 Read로 찾아 question·options(보기 5개)·evidence·과목을 읽어라(정답 없음).

evidence는 **지식그래프 근거**다: 보기에 등장하는 처방들의 [구성약재·주치·계통·출전](한의대 내과학 교과서 기반)과, 질문 증상에 부합하는 처방 후보 목록.
- **처방/변증 문항**: 이 근거를 핵심으로 사용하라 — 질문의 주소증·치법을 보기 처방들의 '주치·구성'과 대조해 가장 부합하는 처방을 고른다.
- evidence가 비었거나 문항과 무관하면 네 한의학 지식으로 답하라(나쁜 근거에 끌려가지 말 것).

과목 특성에 맞게 추론하라:
- 내과학1·2/부인/소아/외과/신경정신/안이비인후: 변증추론(주소증→팔강·장부 변증→치법→보기 처방의 주치·구성 비교→정답).
- 본초학: 본초의 성미·귀경·효능·배합·금기. 침구학: 경혈 위치·소속경락·주치·특정혈. 한약학 응용: 방제 구성(군신좌사)·포제·약전·독성.
- 보건의약관계법규: 정확한 법조문. 예방의학: 역학·보건통계. 한의학기초/한방생리: 원전·기초이론.

단계적으로 보기를 비교·배제해 정답 보기번호(1~5) 하나를 확정하라. {idx:${idx}, pred(1~5)} 출력.`

function chunk(a, n) { const o = []; for (let i = 0; i < a.length; i += n) o.push(a.slice(i, i + n)); return o }
const CH = A.chunk || 8   // 서버 레이트리밋 회피: 순차 청크로 throttle
phase('GraphRAG')
const votes = {}
for (let s = 0; s < K; s++) {
  let done = 0
  for (const grp of chunk(IDXS, CH)) {
    const round = await parallel(grp.map((n) => () =>
      agent(prompt(n, s), { label: `grag-s${s + 1}-${n}`, phase: 'GraphRAG', model: 'opus', schema: SCHEMA }).catch(() => null)
    ))
    for (const r of round) {
      if (!r || r.idx == null || r.pred == null) continue
      ;(votes[r.idx] = votes[r.idx] || []).push(r.pred)
    }
    done += grp.length
    if (done % 80 === 0) log(`샘플${s + 1} 진행 ${done}/${IDXS.length} (응답 ${Object.keys(votes).length})`)
  }
  log(`자기일관성 샘플 ${s + 1}/${K} 완료 (누적 응답 ${Object.keys(votes).length}/${IDXS.length})`)
}
return { votes, meta: { k: K, n: IDXS.length } }
