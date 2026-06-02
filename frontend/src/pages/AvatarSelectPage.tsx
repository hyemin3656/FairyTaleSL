import { useNavigate } from "react-router-dom";
import { AVATAR_OPTIONS, useAvatarStore } from "../stores/avatarStore";
import AvatarPreview from "../components/avatar/AvatarPreview";

export default function AvatarSelectPage() {
  const navigate = useNavigate();
  const { selectedId, setAvatar } = useAvatarStore();

  const handleSelect = (id: string) => {
    setAvatar(id);
    navigate(-1);
  };

  return (
    <div className="avatar-select-page">
      <header className="avatar-select-header">
        <button className="btn-back" onClick={() => navigate(-1)}>← 뒤로</button>
        <h2>아바타 선택</h2>
      </header>

      <div className="avatar-select-grid">
        {AVATAR_OPTIONS.map((avatar) => {
          const isSelected = avatar.id === selectedId;
          return (
            <div
              key={avatar.id}
              className={`avatar-card ${isSelected ? "selected" : ""}`}
              onClick={() => handleSelect(avatar.id)}
            >
              <div className="avatar-card-preview">
                <AvatarPreview url={avatar.url} scale={avatar.sceneScale} />
                {isSelected && <div className="avatar-card-badge">✓ 선택됨</div>}
              </div>
              <div className="avatar-card-info">
                <span className="avatar-card-name">{avatar.name}</span>
                <span className="avatar-card-desc">{avatar.description}</span>
              </div>
              <button
                className={`avatar-card-btn ${isSelected ? "selected" : ""}`}
                onClick={(e) => { e.stopPropagation(); handleSelect(avatar.id); }}
              >
                {isSelected ? "사용 중" : "선택하기"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
