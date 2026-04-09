import http from "./http";

export interface UserOut {
  id: string;
  email: string;
  username: string;
  nickname: string;
  role: string;
  avatar_speed: number;
  subtitle_enabled: boolean;
  created_at: string;
}

export interface SessionQAOut {
  id: string;
  section_order: number;
  question_text: string;
  user_answer_text: string | null;
  llm_response_text: string | null;
  recognition_accuracy: number | null;
  created_at: string;
}

export interface LearningSessionOut {
  id: string;
  book_id: string | null;
  book_title: string | null;
  last_section_order: number;
  status: string;
  avg_recognition_accuracy: number | null;
  started_at: string;
  ended_at: string | null;
  qa_records: SessionQAOut[];
}

export interface MyPageResponse {
  user: UserOut;
  sessions: LearningSessionOut[];
  total_sessions: number;
  completed_sessions: number;
  avg_score: number | null;
}

export async function register(payload: {
  email: string; username: string; nickname: string; password: string;
}): Promise<{ access_token: string }> {
  const { data } = await http.post("/auth/register", payload);
  return data;
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const { data } = await http.post("/auth/login", { username, password });
  return data;
}

export async function fetchMe(): Promise<UserOut> {
  const { data } = await http.get<UserOut>("/auth/me");
  return data;
}

export async function fetchMyPage(): Promise<MyPageResponse> {
  const { data } = await http.get<MyPageResponse>("/sessions/me");
  return data;
}

export async function startSession(bookId?: string): Promise<{ session_id: string }> {
  const { data } = await http.post("/sessions", { book_id: bookId ?? null });
  return data;
}

export async function completeSession(sessionId: string, avgScore: number): Promise<void> {
  await http.patch(`/sessions/${sessionId}`, {
    status: "completed",
    avg_recognition_accuracy: avgScore,
  });
}

export async function saveQARecord(sessionId: string, payload: {
  section_order: number;
  question: string;
  user_answer?: string;
  llm_response?: string;
  score?: number;
}): Promise<void> {
  await http.post(`/sessions/${sessionId}/qa`, payload);
}
