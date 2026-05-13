import http from "./http";

export interface QuizOut {
  id: string;
  book_id: string;
  section_order: number;
  question: string;
}

export interface QuizCheckResponse {
  correct: boolean;
  expected_answer: string | null;   // reveal=true 또는 correct=true 시에만 값
}

export async function fetchBookQuizzes(bookId: string): Promise<QuizOut[]> {
  const { data } = await http.get<QuizOut[]>(`/books/${bookId}/quizzes`);
  return data;
}

export async function fetchSectionQuiz(bookId: string, sectionOrder: number): Promise<QuizOut> {
  const { data } = await http.get<QuizOut>(`/books/${bookId}/quizzes/${sectionOrder}`);
  return data;
}

export async function checkQuizAnswer(
  quizId: string, userAnswer: string, reveal = false
): Promise<QuizCheckResponse> {
  const { data } = await http.post<QuizCheckResponse>(
    `/quizzes/${quizId}/check`,
    { user_answer: userAnswer },
    { params: { reveal } },
  );
  return data;
}
