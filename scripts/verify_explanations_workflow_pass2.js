export const meta = {
  name: '해설-의학정확성-검증-2차강화',
  description: '교정된 해설 517개를 배치당 독립 리뷰어 2명(union)으로 정밀 재검증 후 확정',
  phases: [
    { title: '정밀검증', detail: '배치당 독립 리뷰어 2명(상이 중점) → union' },
    { title: '확정', detail: 'union 플래그를 독립 3차 에이전트로 확정' },
  ],
}

const N = 35
const INDIR = 'D:\\tmp\\verify_in'
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
          source: { type: 'string' }, period: { type: 'number' }, qnum: { type: 'number' }, subject: { type: 'string' },
          severity: { type: 'string', enum: ['경미', '중대'] },
          issue: { type: 'string' }, proposed_correction: { type: 'string' },
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
    is_real: { type: 'boolean' }, final_severity: { type: 'string', enum: ['정확', '경미', '중대'] },
    final_correction: { type: 'string' }, reason: { type: 'string' },
  },
  required: ['is_real', 'final_severity', 'final_correction', 'reason'],
}

const RULES = `이 JSONL 파일을 Read로 읽어라: {PATH}
각 줄 = 한 문항: source, 교시, 과목, 번호, question, options(5지), answer(국시원 공식 정답 1~5), answer_text, 해설.
규칙:
- answer(공식 정답)는 국시원 확정 정답이다. 절대 의심 말 것. 검증 대상은 오직 "해설 문장"의 의학적 사실 정확성.
- severity: "중대"=명백한 사실오류로 학습자를 오도. "경미"=다소 부정확하나 오도까진 아님. 문제없으면 플래그 금지.
- 문체·표현·간결성은 트집잡지 말 것. 오직 의학적 사실 오류만. 보수적으로, 확실한 근거가 있을 때만 플래그.
플래그된 문항만 suspects로 반환(없으면 빈 배열): source, period(=교시 숫자), qnum(=번호 숫자), subject(=과목), severity, issue(구체적), proposed_correction(정확한 서술). n_checked=검증 문항 수. batch_idx={IDX}.`

const verifyPromptA = (b) => `당신은 한의사 국가시험 출제·검토위원 수준의 한의학 전문가다. 해설의 의학적 정확성을 적대적으로 검증하라.
${RULES.replace('{PATH}', b.path).replace('{IDX}', b.idx)}
[중점 A] 특히 **본초(효능·성미·귀경)·방제 구성약물·경락/경혈·포제법**의 사실오류를 집중적으로 본다(단, 다른 영역 오류도 놓치지 말 것).`

const verifyPromptB = (b) => `당신은 독립적인 한의학·보건 전문가다. 해설의 의학적 정확성을 적대적으로 검증하라.
${RULES.replace('{PATH}', b.path).replace('{IDX}', b.idx)}
[중점 B] 특히 **진단 수치·검사치·법규 조문·생리/병리 기전, 그리고 오답을 틀렸다고 한 근거의 논리적 정확성**을 집중적으로 본다(단, 본초·방제 등 다른 영역 오류도 놓치지 말 것).`

const confirmPrompt = (s, b) => `당신은 독립적인 한의학 전문가(최종 판정자)다. 앞선 검토자가 아래 문항의 "해설"에 의학적 오류가 있다고 주장했다. 선입견 없이 처음부터 다시 판단하라 — 주장에 동의할 의무는 없다.
배치 파일을 Read로 읽어라: ${b.path}
대상 문항: source="${s.source}", 교시=${s.period}, 번호=${s.qnum} (과목 ${s.subject || ''}). 이 문항의 question/options/answer/해설을 찾아 정독하라.
검토자 주장: severity=${s.severity} / issue=${s.issue} / 제안교정=${s.proposed_correction}
판단:
- answer는 국시원 공식 정답이니 의심 말 것. "해설" 텍스트가 실제로 의학적 사실오류를 포함하는지만 본다.
- is_real=true는 명백한 사실오류가 실재할 때만. 검토자가 과민반응했고 해설이 옳거나 허용 범위면 false.
- final_severity: 정확/경미/중대. final_correction: is_real=true이고 중대면 드롭인 가능한 교정 전체 해설(3~5문장), 아니면 "". reason: 1~2문장.`

function unionSuspects(ra, rb, b) {
  const map = new Map()
  const ncA = ra && ra.n_checked ? ra.n_checked : 0
  const ncB = rb && rb.n_checked ? rb.n_checked : 0
  const add = (r, who) => {
    if (!r || !Array.isArray(r.suspects)) return
    for (const s of r.suspects) {
      const k = `${s.source}|${s.period}|${s.qnum}`
      if (map.has(k)) {
        const e = map.get(k)
        if (s.severity === '중대') e.severity = '중대'
        e.issue += ` || [${who}] ${s.issue}`
        if (!e.proposed_correction && s.proposed_correction) e.proposed_correction = s.proposed_correction
      } else {
        map.set(k, { source: s.source, period: s.period, qnum: s.qnum, subject: s.subject || '', severity: s.severity, issue: `[${who}] ${s.issue}`, proposed_correction: s.proposed_correction || '' })
      }
    }
  }
  add(ra, 'A'); add(rb, 'B')
  return { n_checked: Math.max(ncA, ncB), suspects: Array.from(map.values()), bothFailed: !ra && !rb }
}

phase('정밀검증')
const unions = await parallel(batches.map((b) => () =>
  parallel([
    () => agent(verifyPromptA(b), { label: `검증A:b${pad(b.idx)}`, phase: '정밀검증', schema: BATCH_SCHEMA }).catch(() => null),
    () => agent(verifyPromptB(b), { label: `검증B:b${pad(b.idx)}`, phase: '정밀검증', schema: BATCH_SCHEMA }).catch(() => null),
  ]).then(([ra, rb]) => ({ b, u: unionSuspects(ra, rb, b) }))
))
const failed = unions.filter((x) => x.u.bothFailed).map((x) => x.b.idx)
const checked = unions.reduce((a, x) => a + (x.u.n_checked || 0), 0)
const jobs = []
unions.forEach((x) => x.u.suspects.forEach((s) => jobs.push({ s, b: x.b })))
log(`정밀검증: 검증 ${checked}문항 · union 플래그 ${jobs.length}건 · 양쪽실패 ${failed.length}`)

phase('확정')
let confirmed = []
if (jobs.length) {
  const confirms = await parallel(jobs.map(({ s, b }) => () =>
    agent(confirmPrompt(s, b), { label: `확정:${s.source}-${s.period}-${s.qnum}`, phase: '확정', schema: CONFIRM_SCHEMA })
      .then((v) => ({ source: s.source, period: s.period, qnum: s.qnum, subject: s.subject || '', union_issue: s.issue, ...v }))
      .catch(() => null)
  ))
  confirmed = confirms.filter(Boolean).filter((x) => x.is_real)
}
log(`확정: 신규/잔존 오류 ${confirmed.length}건 (중대 ${confirmed.filter((c) => c.final_severity === '중대').length})`)

return { checked, flagged: jobs.length, confirmed, failed_batches: failed }
