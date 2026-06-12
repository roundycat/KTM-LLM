export const meta = {
  name: '해설-의학정확성-검증',
  description: '한의학 해설 517개를 다중 에이전트로 적대적 의학정확성 검증 후 확정 오류 도출',
  phases: [
    { title: '검증', detail: '35배치 각각 적대적 1차 검토(의학 사실오류)' },
    { title: '재검증', detail: '플래그 항목을 독립 2차 검토로 확정(거짓양성 제거)' },
  ],
}

const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const N = A.count || 35
const INDIR = A.indir || 'D:\\tmp\\verify_in'
const pad = (i) => String(i).padStart(3, '0')
const batches = Array.from({ length: N }, (_, i) => ({ idx: i, path: `${INDIR}\\vbatch_${pad(i)}.jsonl` }))

const BATCH_SCHEMA = {
  type: 'object',
  properties: {
    batch_idx: { type: 'number' },
    n_checked: { type: 'number' },
    suspects: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          source: { type: 'string' },
          period: { type: 'number', description: '교시' },
          qnum: { type: 'number', description: '번호' },
          subject: { type: 'string', description: '과목' },
          severity: { type: 'string', enum: ['경미', '중대'] },
          issue: { type: 'string', description: '무엇이 왜 의학적으로 틀렸는지 구체적으로' },
          proposed_correction: { type: 'string', description: '정확한 서술' },
        },
        required: ['source', 'period', 'qnum', 'severity', 'issue', 'proposed_correction'],
      },
    },
  },
  required: ['batch_idx', 'n_checked', 'suspects'],
}

const CONFIRM_SCHEMA = {
  type: 'object',
  properties: {
    is_real: { type: 'boolean', description: '실제 의학적 사실오류가 맞으면 true' },
    final_severity: { type: 'string', enum: ['정확', '경미', '중대'] },
    final_correction: { type: 'string', description: '확정 중대오류면 드롭인 가능한 교정 전체 해설(3~5문장), 아니면 빈 문자열' },
    reason: { type: 'string', description: '판단 근거 1~2문장' },
  },
  required: ['is_real', 'final_severity', 'final_correction', 'reason'],
}

const verifyPrompt = (b) => `당신은 한의사 국가시험 출제·검토위원 수준의 한의학 전문가다. 적대적(critical) 2차 검토자처럼, 주어진 "해설"의 의학적 정확성을 엄격히 검증하라.

이 JSONL 파일을 Read로 읽어라: ${b.path}
각 줄 = 한 문항: source, 교시, 과목, 번호, question, options(5지선다), answer(국시원 공식 정답 번호 1~5), answer_text, 해설.

검증 규칙:
- answer(공식 정답)는 국시원이 확정한 정답이다. 절대 의심하지 말 것. 너의 임무는 정답 자체가 아니라 "해설 문장"의 의학적 사실 정확성 검증이다.
- 점검: (1) 정답이 옳은 이유 서술이 의학적으로 정확한가, (2) 본초의 효능·성미·귀경, 경락·경혈 위치/소속, 방제 구성약물, 법규 조문, 생리·병리 기전 등에 명백한 사실오류가 있는가, (3) 오답을 틀렸다고 한 근거가 정확한가.
- severity: "중대" = 명백한 사실오류로 학습자를 오도함. "경미" = 다소 부정확하나 오도까진 아님. 문제 없으면 플래그하지 말 것.
- 문체·표현·간결성·생략은 트집잡지 말 것. 오직 의학적 사실 오류만.
- 보수적으로 판단: 확실한 의학적 근거가 있을 때만 플래그하라. 애매하면 플래그하지 마라.

플래그된 문항만 suspects 배열로 반환하라(없으면 빈 배열). 각 suspect는 source, period(=교시 숫자), qnum(=번호 숫자), subject(=과목), severity, issue(구체적 오류 설명), proposed_correction(정확한 서술)을 담는다.
n_checked = 이 파일에서 검증한 문항 수. batch_idx = ${b.idx}.`

const confirmPrompt = (s, b) => `당신은 독립적인 한의학 전문가(별도 2차 검토자)다. 1차 검토자가 아래 문항의 "해설"에 의학적 오류가 있다고 주장했다. 선입견 없이 처음부터 다시 판단하라 — 1차 주장에 동의할 의무는 전혀 없다.

이 배치 파일을 Read로 읽어라: ${b.path}
대상 문항 식별 → source="${s.source}", 교시=${s.period}, 번호=${s.qnum} (과목 ${s.subject || ''}). 이 문항의 question/options/answer/해설을 파일에서 찾아 정독하라.

1차 검토자의 주장:
- severity: ${s.severity}
- issue: ${s.issue}
- 제안 교정: ${s.proposed_correction}

판단 지침:
- answer는 국시원 공식 정답이므로 의심하지 말 것. "해설" 텍스트가 실제로 의학적 사실오류를 포함하는지만 본다.
- is_real=true 는 명백한 의학적 사실오류가 실재할 때만. 1차가 과민반응했고 해설이 실제로 옳거나 허용 범위면 is_real=false.
- final_severity: 정확 / 경미 / 중대.
- final_correction: is_real=true 이고 중대일 때만, 그 문항의 교정된 "전체 해설"(3~5문장, 한국어, 정답 근거 + 핵심 오답 분석)을 드롭인 가능하게 작성. 그 외에는 빈 문자열.
- reason: 판단 근거 1~2문장.`

// ── 1차: 35배치 적대적 검증 (배리어: 전체 suspect 수집 + 실패배치 커버리지 확인 필요) ──
phase('검증')
const reviews = await parallel(
  batches.map((b) => () =>
    agent(verifyPrompt(b), { label: `검증:b${pad(b.idx)}`, phase: '검증', schema: BATCH_SCHEMA }).catch(() => null)
  )
)
const failed = batches.filter((b, i) => !reviews[i]).map((b) => b.idx)
const checked = reviews.filter(Boolean).reduce((a, r) => a + (r.n_checked || 0), 0)
const suspectJobs = []
reviews.forEach((r, i) => {
  if (r && Array.isArray(r.suspects)) r.suspects.forEach((s) => suspectJobs.push({ s, b: batches[i] }))
})
log(`1차 완료: 검증 ${checked}문항 · 플래그 ${suspectJobs.length}건 · 실패배치 ${failed.length}`)

// ── 2차: 플래그 항목 독립 재검증 (거짓양성 제거) ──
phase('재검증')
let confirmed = []
if (suspectJobs.length) {
  const confirms = await parallel(
    suspectJobs.map(({ s, b }) => () =>
      agent(confirmPrompt(s, b), { label: `재검증:${s.source}-${s.period}-${s.qnum}`, phase: '재검증', schema: CONFIRM_SCHEMA })
        .then((v) => ({ source: s.source, period: s.period, qnum: s.qnum, subject: s.subject || '', first_severity: s.severity, first_issue: s.issue, ...v }))
        .catch(() => null)
    )
  )
  confirmed = confirms.filter(Boolean).filter((x) => x.is_real)
}
log(`2차 완료: 확정 오류 ${confirmed.length}건 (중대 ${confirmed.filter((c) => c.final_severity === '중대').length})`)

return {
  checked,
  flagged: suspectJobs.length,
  confirmed,
  failed_batches: failed,
}
