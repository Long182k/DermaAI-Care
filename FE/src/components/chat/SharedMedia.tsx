import { useEffect, useState } from "react";
import { useChatContext } from "stream-chat-react";

export default function SharedMedia() {
  const { channel } = useChatContext();
  const [sharedMedia, setSharedMedia] = useState<string[]>([]);

  useEffect(() => {
    if (!channel) return;

    const subscription = channel.on("message.new", (event) => {
      event.message.attachments?.forEach((attachment) => {
        if (attachment.type === "image" && attachment.image_url) {
          setSharedMedia((prev) => [...prev, attachment.image_url]);
        }
      });
    });

    return () => {
      subscription.unsubscribe();
    };
  }, [channel]);

  useEffect(() => {
    // Add optional chaining to safely access nested properties
    if (!channel?.state?.messages) return;

    const imageAttachments = channel.state.messages
      .flatMap((message) =>
        (message.attachments || []).filter(
          (att) => att.type === "image" && att.image_url
        )
      )
      .map((att) => att.image_url);

    setSharedMedia(imageAttachments);
  }, [channel]);

  return (
    <div className="w-[500px] border-l border-gray-200 bg-white p-4 ">
      <h2 className="text-xl font-semibold mb-4">Shared Media</h2>
      <div className="grid grid-cols-2 gap-2">
        {sharedMedia?.map((image, index) => (
          <div
            key={index}
            className="overflow-hidden rounded-lg w-[150px] h-[150px]"
          >
            <img
              src={image || "/placeholder.svg"}
              alt={`Shared image ${index + 1}`}
              className="object-cover w-full h-full"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
