import { doctorApi } from "@/api/appointment";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ChannelListMessengerProps, useChatContext } from "stream-chat-react";
import { Button } from "../ui/button";
import { capitalizeWords } from "@/@util/helpers";

export function ChannelListCustom({
  loadedChannels,
}: ChannelListMessengerProps) {
  const { setActiveChannel, channel: activeChannel } = useChatContext();
  const { client } = useChatContext();

  const createChannel = useMutation({
    mutationFn: ({
      name,
      memberIds,
      imageUrl,
    }: {
      name: string;
      memberIds: string[];
      imageUrl?: string;
    }) => {
      if (client == null) throw Error("Not connected");

      return client
        .channel("messaging", crypto.randomUUID(), {
          name,
          image: imageUrl,
          members: [client.userID, ...memberIds],
        })
        .create();
    },
    onSuccess() {
      setIsModalOpen(false);
    },
  });

  const {
    data: doctors,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["doctors"],
    queryFn: () => doctorApi.getAllDoctors(),
    select: (res) => res.doctors,
  });

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedDoctorId, setSelectedDoctorId] = useState<
    string | undefined
  >();

  const handleNewConversationClick = () => {
    setIsModalOpen(true);
  };

  const handleStartConversation = () => {
    const selectedDoctor = doctors.find((d) => d.id === selectedDoctorId);

    if (selectedDoctorId) {
      createChannel.mutate({
        name: capitalizeWords(
          `Dr. ${selectedDoctor?.firstName} ${selectedDoctor?.lastName}`
        ),
        imageUrl: selectedDoctor?.avatarUrl,
        memberIds: [selectedDoctorId],
      });
      setIsModalOpen(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 m-3 h-full">
      <Button
        onClick={handleNewConversationClick}
        className="text-xl font-bold text-center"
      >
        New Conversation
      </Button>

      {isModalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 shadow-lg w-80">
            <h2 className="text-lg font-semibold mb-4">Select a Doctor</h2>
            {isLoading ? (
              <p>Loading...</p>
            ) : error ? (
              <p>Error loading doctors</p>
            ) : (
              <select
                value={selectedDoctorId}
                onChange={(e) => setSelectedDoctorId(e.target.value)}
                className="w-full p-2 border rounded mb-4"
              >
                <option value="">-- Choose a doctor --</option>
                {doctors
                  ?.filter((d) => {
                    const channelsName = loadedChannels?.map(
                      (c) => c.data?.name
                    );
                    return !channelsName?.includes(
                      capitalizeWords(`Dr. ${d?.firstName} ${d?.lastName}`)
                    );
                  })
                  ?.map((doctor) => (
                    <option key={doctor.id} value={doctor.id}>
                      {doctor.firstName} {doctor.lastName || doctor.userName}
                    </option>
                  ))}
              </select>
            )}

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsModalOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleStartConversation}
                disabled={!selectedDoctorId}
              >
                Start
              </Button>
            </div>
          </div>
        </div>
      )}

      <hr className="border-gray-500" />

      {loadedChannels != null && loadedChannels.length > 0
        ? loadedChannels.map((channel) => {
            const isActive = channel === activeChannel;
            const extraClasses = isActive
              ? "bg-accent text-white"
              : "cursor-pointer hover:bg-accent";
            
            // Find the opposite user in the channel members
            const getOppositeUser = () => {
              if (!channel?.state?.members) return null;
              
              // Find the member that is not the current user
              const oppositeUserEntry = Object.entries(channel.state.members).find(
                ([_, memberData]) => memberData.user?.id !== client.userID
              );
              
              return oppositeUserEntry ? oppositeUserEntry[1].user : null;
            };
            
            const oppositeUser = getOppositeUser();

            return (
              <button
                onClick={() => setActiveChannel(channel)}
                disabled={isActive}
                className={`flex items-center gap-3 p-3 rounded-lg ${extraClasses}`}
                key={channel.id}
              >
                <div className="relative">
                  <img
                    src={oppositeUser?.image || "/placeholder.svg"}
                    alt="avatar"
                    className="w-12 h-12 rounded-full object-cover"
                  />
                  {/* Online Dot */}
                  <span className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></span>
                </div>

                <div className="flex flex-col items-start">
                  <span className="text-sm font-medium text-black">
                    {oppositeUser?.name || "User"}
                  </span>
                  <span className="text-xs text-gray-500 uppercase">
                    DOCTOR
                  </span>
                </div>
              </button>
            );
          })
        : "No Conversations"}
    </div>
  );
}
