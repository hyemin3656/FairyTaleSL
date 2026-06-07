import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface AvatarOption {
  id: string;
  name: string;
  url: string;
  preview: string;
  description: string;
  sceneScale?: number;    // 동화 읽기 화면 스케일 (기본 1.0)
  sceneOffsetY?: number;  // 동화 읽기 화면 추가 Y 오프셋 (기본 0)
  previewScale?: number;  // 아바타 선택 페이지 미리보기 스케일 (기본 sceneScale)
}

export const AVATAR_OPTIONS: AvatarOption[] = [
  {
    id: "default",
    name: "성준",
    url: "/avatar.glb",
    preview: "/avatars/preview_default.png",
    description: "씩씩하고 용감한 남자 아바타",
    sceneScale: 0.7,
    sceneOffsetY: -0.1,
  },
  {
    id: "avatar2",
    name: "동순",
    url: "/avatar2.glb",
    preview: "/avatars/preview2.png",
    description: "밝고 활발한 여자 아바타",
    sceneScale: 1.0,
    sceneOffsetY: -0.15,
  },
  {
    id: "avatar3",
    name: "혜미",
    url: "/avatar3.glb",
    preview: "/avatars/preview3.png",
    description: "차분하고 친근한 여자 아바타",
    sceneScale: 0.68,
    sceneOffsetY: -0.05,
    previewScale: 0.5,
  },
];

interface AvatarState {
  selectedId: string;
  setAvatar: (id: string) => void;
  selectedAvatar: () => AvatarOption;
}

export const useAvatarStore = create<AvatarState>()(
  persist(
    (set, get) => ({
      selectedId: "default",
      setAvatar: (id) => set({ selectedId: id }),
      selectedAvatar: () =>
        AVATAR_OPTIONS.find((a) => a.id === get().selectedId) ?? AVATAR_OPTIONS[0],
    }),
    { name: "fairytale-avatar" }
  )
);
