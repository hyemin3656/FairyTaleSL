/**
 * scenarioStore — FairyTaleSL 학습 시나리오 상태 머신 (Zustand)
 *
 * 전이도:
 *   IDLE
 *    └─[startSession]→ STORY_PLAYING ⇄ [pause/resume] ⇄ PAUSED
 *                       │                │
 *                       │                └─[enterChildQuestion]→ CHILD_QUESTION
 *                       │                                          └─[exitChildQuestion]→ PAUSED→(auto resume)
 *                       └─[sectionEnd]→ SECTION_DONE
 *                                        ├─[enterFollowAlong]→ FOLLOW_ALONG ─[passFollow/skipFollow]→ SECTION_DONE
 *                                        └─[startQuiz]→ QUIZ ─[finishQuiz]→ nextSection() | complete()
 *                                                                              │            │
 *                                                                       STORY_PLAYING   COMPLETED
 *
 * 부 모드(따라하기/질문하기)는 선택, 주 모드(퀴즈)는 섹션 종료마다 필수.
 * 잘못된 전이는 console.warn 후 noop.
 */
import { create } from "zustand";
import type { QuizOut } from "../api/quizzes";

export type ScenarioMode =
  | "IDLE"
  | "STORY_PLAYING"
  | "PAUSED"
  | "SECTION_DONE"
  | "FOLLOW_ALONG"
  | "CHILD_QUESTION"
  | "QUIZ"
  | "COMPLETED";

// 클라이언트는 expected_answer/synonyms를 알 수 없다. 매칭은 서버 책임.
export type Quiz = QuizOut;

export interface SectionResult {
  sectionIdx: number;
  followAlongPassed: boolean | null;   // null = 미시도/스킵
  quizCorrect: boolean | null;         // null = 정답보기 선택
  quizAttempts: number;                // 0 = 정답보기, 1~MAX = 풀기 시도 횟수
}

interface ScenarioState {
  mode: ScenarioMode;

  // 컨텍스트
  bookId: string | null;
  sessionId: string | null;
  sectionIdx: number;
  totalSections: number;

  // PAUSED→CHILD_QUESTION→PAUSED 복원용 토큰 인덱스
  pausedTokenIndex: number | null;

  // 모드 내 데이터
  followAlongTarget: string | null;
  pendingQuiz: Quiz | null;

  // 결과 누적
  results: SectionResult[];

  // ── 전이 액션 ──────────────────────────────────────────────────────────
  startSession: (args: { bookId: string; sessionId: string; totalSections: number }) => void;
  pause: (tokenIndex: number) => void;
  resume: () => void;
  sectionEnd: () => void;
  skipPlayback: () => void;     // 아바타 재생을 건너뛰고 곧장 SECTION_DONE

  enterFollowAlong: (gloss: string) => void;
  passFollowAlong: () => void;     // 성공 → 결과 기록 + SECTION_DONE
  skipFollowAlong: () => void;     // 스킵 → 결과 null + SECTION_DONE

  enterChildQuestion: () => void;
  exitChildQuestion: () => void;   // → PAUSED (호출자가 즉시 resume 가능)

  startQuiz: (quiz: Quiz) => void;
  finishQuiz: (result: { correct: boolean | null; attempts: number }) => void;
  cancelQuiz: () => void;        // 결과 기록 없이 QUIZ → SECTION_DONE 복귀

  nextSection: () => void;
  prevSection: () => void;       // 이전 섹션으로 되돌아가기 (sectionIdx ≥ 1일 때)
  complete: () => void;
  reset: () => void;
}

const initial: Omit<ScenarioState,
  | "startSession" | "pause" | "resume" | "sectionEnd" | "skipPlayback"
  | "enterFollowAlong" | "passFollowAlong" | "skipFollowAlong"
  | "enterChildQuestion" | "exitChildQuestion"
  | "startQuiz" | "finishQuiz" | "cancelQuiz"
  | "nextSection" | "prevSection" | "complete" | "reset"
> = {
  mode: "IDLE",
  bookId: null,
  sessionId: null,
  sectionIdx: 0,
  totalSections: 0,
  pausedTokenIndex: null,
  followAlongTarget: null,
  pendingQuiz: null,
  results: [],
};

function warnInvalid(action: string, current: ScenarioMode) {
  // eslint-disable-next-line no-console
  console.warn(`[scenario] ${action} ignored in mode=${current}`);
}

function ensureResultSlot(results: SectionResult[], idx: number): SectionResult[] {
  if (results.some((r) => r.sectionIdx === idx)) return results;
  return [...results, { sectionIdx: idx, followAlongPassed: null, quizCorrect: null, quizAttempts: 0 }];
}

function updateResult(
  results: SectionResult[], idx: number, patch: Partial<SectionResult>
): SectionResult[] {
  const ensured = ensureResultSlot(results, idx);
  return ensured.map((r) => (r.sectionIdx === idx ? { ...r, ...patch } : r));
}

export const useScenarioStore = create<ScenarioState>((set, get) => ({
  ...initial,

  startSession: ({ bookId, sessionId, totalSections }) => {
    set({
      ...initial,
      bookId, sessionId, totalSections,
      mode: "STORY_PLAYING",
    });
  },

  pause: (tokenIndex) => {
    const { mode } = get();
    if (mode !== "STORY_PLAYING") return warnInvalid("pause", mode);
    set({ mode: "PAUSED", pausedTokenIndex: tokenIndex });
  },

  resume: () => {
    const { mode } = get();
    if (mode !== "PAUSED") return warnInvalid("resume", mode);
    set({ mode: "STORY_PLAYING" });
    // pausedTokenIndex는 useGlossWS resume() 측에서 읽고 처리 후 자체 정리
  },

  sectionEnd: () => {
    const { mode, sectionIdx, results } = get();
    if (mode !== "STORY_PLAYING") return warnInvalid("sectionEnd", mode);
    set({
      mode: "SECTION_DONE",
      results: ensureResultSlot(results, sectionIdx),
    });
  },

  skipPlayback: () => {
    const { mode, sectionIdx, results } = get();
    if (mode !== "STORY_PLAYING" && mode !== "PAUSED") {
      return warnInvalid("skipPlayback", mode);
    }
    set({
      mode: "SECTION_DONE",
      pausedTokenIndex: null,
      results: ensureResultSlot(results, sectionIdx),
    });
  },

  enterFollowAlong: (gloss) => {
    const { mode } = get();
    if (mode !== "SECTION_DONE") return warnInvalid("enterFollowAlong", mode);
    set({ mode: "FOLLOW_ALONG", followAlongTarget: gloss });
  },

  passFollowAlong: () => {
    const { mode, sectionIdx, results } = get();
    if (mode !== "FOLLOW_ALONG") return warnInvalid("passFollowAlong", mode);
    set({
      mode: "SECTION_DONE",
      followAlongTarget: null,
      results: updateResult(results, sectionIdx, { followAlongPassed: true }),
    });
  },

  skipFollowAlong: () => {
    const { mode, sectionIdx, results } = get();
    if (mode !== "FOLLOW_ALONG") return warnInvalid("skipFollowAlong", mode);
    set({
      mode: "SECTION_DONE",
      followAlongTarget: null,
      results: updateResult(results, sectionIdx, { followAlongPassed: false }),
    });
  },

  enterChildQuestion: () => {
    const { mode } = get();
    if (mode !== "PAUSED") return warnInvalid("enterChildQuestion", mode);
    set({ mode: "CHILD_QUESTION" });
  },

  exitChildQuestion: () => {
    const { mode } = get();
    if (mode !== "CHILD_QUESTION") return warnInvalid("exitChildQuestion", mode);
    set({ mode: "PAUSED" });
    // 호출자(BookReadPage)가 즉시 resume() 호출 → token_index 복원
  },

  startQuiz: (quiz) => {
    const { mode } = get();
    if (mode !== "SECTION_DONE") return warnInvalid("startQuiz", mode);
    set({ mode: "QUIZ", pendingQuiz: quiz });
  },

  finishQuiz: ({ correct, attempts }) => {
    const { mode, sectionIdx, results } = get();
    if (mode !== "QUIZ") return warnInvalid("finishQuiz", mode);
    set({
      mode: "SECTION_DONE",
      pendingQuiz: null,
      results: updateResult(results, sectionIdx, {
        quizCorrect: correct,
        quizAttempts: attempts,
      }),
    });
  },

  cancelQuiz: () => {
    const { mode } = get();
    if (mode !== "QUIZ") return warnInvalid("cancelQuiz", mode);
    set({ mode: "SECTION_DONE", pendingQuiz: null });
  },

  nextSection: () => {
    const { mode, sectionIdx, totalSections } = get();
    if (mode !== "SECTION_DONE") return warnInvalid("nextSection", mode);
    const nextIdx = sectionIdx + 1;
    if (nextIdx >= totalSections) {
      set({ mode: "COMPLETED" });
    } else {
      set({ mode: "STORY_PLAYING", sectionIdx: nextIdx, pausedTokenIndex: null });
    }
  },

  // 이전 섹션으로 되돌아가기. 학습 흐름의 중간(STORY_PLAYING/PAUSED/SECTION_DONE)에서만
  // 허용 — QUIZ/CHILD_QUESTION/FOLLOW_ALONG 중에는 그 모드를 먼저 종료해야 한다.
  prevSection: () => {
    const { mode, sectionIdx } = get();
    if (mode !== "STORY_PLAYING" && mode !== "PAUSED" && mode !== "SECTION_DONE") {
      return warnInvalid("prevSection", mode);
    }
    if (sectionIdx <= 0) return;   // 첫 섹션에선 무시
    set({
      mode: "STORY_PLAYING",
      sectionIdx: sectionIdx - 1,
      pausedTokenIndex: null,
      followAlongTarget: null,
      pendingQuiz: null,
    });
  },

  complete: () => {
    set({ mode: "COMPLETED" });
  },

  reset: () => set({ ...initial }),
}));

// ── Selectors ────────────────────────────────────────────────────────────
export const selectCanEnterFollowAlong = (s: ScenarioState) =>
  s.mode === "SECTION_DONE" && s.results.find((r) => r.sectionIdx === s.sectionIdx)?.followAlongPassed === null;

export const selectCanEnterChildQuestion = (s: ScenarioState) => s.mode === "PAUSED";

export const selectQuizRequired = (s: ScenarioState) =>
  s.mode === "SECTION_DONE" &&
  (s.results.find((r) => r.sectionIdx === s.sectionIdx)?.quizCorrect ?? -1) === null &&
  (s.results.find((r) => r.sectionIdx === s.sectionIdx)?.quizAttempts ?? 0) === 0;

export const selectOverallProgress = (s: ScenarioState) => ({
  done: s.results.filter((r) => r.quizCorrect !== null || r.quizAttempts > 0).length,
  total: s.totalSections,
});
